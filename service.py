import asyncio
import json

import signal

from tornado.options import parse_command_line, define
import uvloop

import torchvision.models as models

from classifier import ImageClassifier
from web_app import make_app


class PredicateService(object):
    def __init__(self):
        self.web_app = None
        self.model = None

        # close your eyes and go forward
        # TODO: move to constants or options
        with open('imagenet_class_index.json', 'r') as f:
            self.classes_map = json.load(f)

        self._define_options()

        self.options = self._parse_options()

        self._setup_signal_handlers()

        self._initialize()

    @classmethod
    def _define_options(cls):
        define('host', type=str, default='127.0.0.1')
        define('port', type=int, default=8000)

        define('debug', type=bool, default=False)

    @classmethod
    def _parse_options(cls):
        # it's igly but I don't want to define tornado options in global space
        from tornado.options import options as tornado_options
        parse_command_line()
        return tornado_options

    def _shutdown_service(self):
        loop = asyncio.get_running_loop()

        for task in asyncio.all_tasks():
            task.cancel()

        loop.stop()

    def _setup_signal_handlers(self):
        loop = asyncio.get_event_loop()

        loop.add_signal_handler(signal.SIGINT, self._shutdown_service)
        loop.add_signal_handler(signal.SIGTERM, self._shutdown_service)

    def _initialize_web_app(self):
        return make_app(self, self.options)

    def _initialize_model(self):
        # freeze the loop in first time
        model = models.alexnet(pretrained=True)
        model.eval()

        return model

    def _initialize(self):
        self.web_app = self._initialize_web_app()
        self.model = self._initialize_model()

    async def process_image(self, image_bytes):
        loop = asyncio.get_running_loop()

        classifier = ImageClassifier(self.model, self.classes_map)

        # run in separate process, GIL will work, but in future we can change it for another executor
        result = await loop.run_in_executor(None, classifier.classify, image_bytes)

        prod, class_ = result

        if not class_:
            return {'result': None}

        return {'result': class_[1]}

    def run(self):
        self.web_app.listen(self.options.port, address=self.options.host)


def main():
    # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    loop = asyncio.get_event_loop()

    service = PredicateService()

    service.run()

    try:
        loop.run_forever()
    finally:
        loop.close()


if __name__ == '__main__':
    main()