This is my final O-ring checker for the Assignment.
It checks each image and gives a PASS or FAIL result.
I use NumPy for the image logic.
I only use OpenCV to read/save images and draw text.

Run:
`python src/main.py`

Debug mode:
`python src/main.py --debug`

Results are saved in the `output` folder.

Issues:
I wasn't able to get Oring 15 to fail because when I made it stricter because it was failing good rings like Oring 6 and Oring 13 as well.
But I got the other bad Orings to fail like 5 and 9, but making it any stricter resulted in failing good Orings.
