# ocr

This repository has me trying and researching on how to extract text and draw bounding box on a scanned document which contains both printed and handwritten text, I have currently tried the following:
1. Paddle ocr
2. Pytesseract
3. Keras ocr
4. Easy ocr

Out of these paddle ocr performed the worst with mostly recoginizing printed text only.

Pytesseract on the other hand had good text extraction on almost perfect on printed text while somewhat okay on handwritten text but pytesseract had the worst bounding box coordinates that were way off their original place.

Keras and Easy ocr both performed okay on both the fields of text extraction and bounding box coordinates

I have uploaded all the results I have produced till yet and I will continue to do so while I am trying other OCR and VDU techniques and modules.
