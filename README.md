### How to run

1) Create a new virtual env (or use already existing)
2) Install dependencies 
```bash
pip install -r requiremnts.txt
```
3) Run it
```
python service.py
```

It will run a web server on `8000` port by default.

### DEPENDENCIES 
Only python 3.7 - because I had used some new method from it:
 * `asyncio.get_running_loop()`
 * `asyncio.all_tasks`