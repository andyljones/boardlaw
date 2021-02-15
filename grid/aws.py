# * Rent an EC2 instance
# * Install docker: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html
# ```
# sudo amazon-linux-extras install docker -y
# sudo service docker start
# sudo usermod -a -G docker ec2-user
# ```
# Log out and in, then `docker info` to verify.
# ```
# docker run andyljones/boardlaw --network host --name boardlaw
# ```