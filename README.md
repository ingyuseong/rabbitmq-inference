# RabbitMQ Inference
A repo for implement a simple message queue based server architecture to asynchronously handle resource-intensive tasks(e.g. ML inference). For detailed information about RabbitMQ, please check this article: [Handling resource-intensive tasks with work queues in RabbitMQ](https://www.cloudamqp.com/blog/work-queues-in-rabbitmq-for-resource-intensive-tasks.html).

An abstractive running process of this code would be as following:
1. Subscribe a request message from the API Server (request queue)
    * In this example, the API Server is implemented in `node.js`
3. StarGAN v2 Inference: Generate 8 images and upload them to AWS S3
    * The base code was derived from [StarGAN v2](https://github.com/clovaai/stargan-v2)
4. Publish a result message to the subscriber on the API Server (result queue)

## Requirements
* Linux is recommended for performance and compatibility reasons.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA toolkit and cuDNN.
* 64-bit Python 3.6 installation. I strongly recommend Anaconda3.
* Install the packages from `requirements.txt`.
* Docker users: use the provided `Dockerfile` to build an image with the required library dependencies.
