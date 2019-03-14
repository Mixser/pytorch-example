import logging


from tornado.httpclient import AsyncHTTPClient, HTTPError
from tornado.web import Application, RequestHandler, url

from tornado import escape


logger = logging.getLogger(__name__)


class PredicateHandler(RequestHandler):
    async def post(self):
        try:
            data = escape.json_decode(self.request.body)
        except (TypeError, ValueError) as exc:
            return self.write({'error': str(exc)})

        if 'url' not in data:
            return self.write({'error': "You must specify the url of image"})

        url = data['url']

        result = await self.application.process_url(url)

        return self.write(result)


class WebApplication(Application):
    def __init__(self, service, *args, **kwargs):
        self.service = service

        self._http_client = AsyncHTTPClient()

        super(WebApplication, self).__init__(*args, **kwargs)

    async def process_url(self, url):
        try:
            response = await self._http_client.fetch(url)
        except (OSError, HTTPError) as exc:
            return {'error': str(exc)}

        try:
            result = await self.service.process_image(response.body)
        except Exception as exc:
            logger.error('[PROCESS URL ERROR] url: %s', url, exc_info=exc)
            result = {'error': "Can't process the image."}

        return result


def make_app(service, options):

    app = WebApplication(service,
                         [url(r'/predict', PredicateHandler)],
                         **options.as_dict())

    return app


__all__ = [
    'make_app'
]