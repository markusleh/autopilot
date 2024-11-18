export PATH=$PATH:/Applications/FlightGear.app/Contents/MacOS/
shaping=STANDARD
task=$2
env_id=C172-$task-Shaping.$shaping-NoFG-v0
if [ "$1" == "train" ]; then
    python ppo.py --env-id=$env_id --no-capture-video --total-timesteps=1000000 
fi
if [ "$1" == "eval" ]; then
    env_id=C172-$task-Shaping.$shaping-FG-v0
    python eval-only.py --env-id=$env_id --capture-video
fi