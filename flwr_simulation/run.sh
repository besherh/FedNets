echo "Starting server"
python server_qFedAvg.py &
#python server.py &

sleep 3  # Sleep for 3s to give the server enough time to start
#clinet_ids=('0' '1' '10' '11' '12' '13' '14' '15' '16' '17')
clinet_ids=('0' '1')

for i in "${clinet_ids[@]}"
do
    echo "Starting client $i"
    python client.py --partition=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait