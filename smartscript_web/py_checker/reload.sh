sleep 2
sudo supervisorctl restart smart-scripts
sleep 3
sudo chmod 777 /home/config/smart-scripts.sock
sudo service nginx restart
