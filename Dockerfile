FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "gradio>=5,<6" pillow

COPY model.py app.py ./
COPY examples ./examples
COPY results ./results

ENV IN_DOCKER=1
EXPOSE 7860
CMD ["python", "app.py"]
