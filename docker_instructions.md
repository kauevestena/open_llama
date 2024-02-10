docker build --tag 'open_llama' .

docker run --name running_open_llama -v D:\llama:/workspace/data --gpus all -it 'open_llama' 