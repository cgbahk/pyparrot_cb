"""
Demo the direct flying for the python interface

Author: Amy McGovern
"""

from pyparrot.Minidrone import Mambo

# you will need to change this to the address of YOUR mambo
mamboAddr = "e0:14:d0:63:3d:d0"

# make my mambo object
# remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
mambo = Mambo(mamboAddr, use_wifi=True)

print("trying to connect")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if (success):
    # get the state information
    print("sleeping")
    mambo.smart_sleep(2)
    mambo.ask_for_state_update()
    mambo.smart_sleep(2)

    print("taking off!")
    mambo.safe_takeoff(5)

    print("Flying direct: going up")
    mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=50, duration=0.5)
    mambo.smart_sleep(2)


    print("Flying direct: going forward (positive pitch)")
    for i in range(20):
        mambo.fly_direct(roll=0, pitch=-10.5, yaw=0, vertical_movement=0, duration=0.1)
        mambo.smart_sleep(0.01)
    mambo.smart_sleep(5)
    for i in range(20):
        mambo.fly_direct(roll=0, pitch=+10.51, yaw=0, vertical_movement=0, duration=0.1)
        mambo.smart_sleep(0.01)
    mambo.smart_sleep(5)

    # print("Showing turning (in place) using turn_degrees")
    # for i in range(10):
    #     mambo.turn_degrees(45)
    #     print(">")
    #     # mambo.smart_sleep(0.3)
    #     mambo.turn_degrees(-45)
    #     print("<")
    #     # mambo.smart_sleep(0.3)


    # print("Flying direct: yaw")
    # for i in range(20):
    #     mambo.fly_direct(roll=0, pitch=0, yaw=50, vertical_movement=0, duration=0.1)
    #     # mambo.smart_sleep(0.01)
    # mambo.smart_sleep(5)
    # for i in range(10):
    #     mambo.fly_direct(roll=0, pitch=0, yaw=-50, vertical_movement=0, duration=0.1)
    #     # mambo.smart_sleep(0.01)
    # mambo.smart_sleep(5)

    # print("Flying direct: going backwards (negative pitch)")
    # mambo.fly_direct(roll=0, pitch=-50, yaw=0, vertical_movement=0, duration=0.5)

    # print("Flying direct: roll")
    # for i in range(20):
    #     mambo.fly_direct(roll=50, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
    #     mambo.smart_sleep(0.01)
    # mambo.smart_sleep(5)
    # for i in range(20):
    #     mambo.fly_direct(roll=-50, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
    #     mambo.smart_sleep(0.01)
    # mambo.smart_sleep(5)

    # print("Flying direct: going up")
    # mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=50, duration=1)
    #
    # print("Flying direct: going around in a circle (yes you can mix roll, pitch, yaw in one command!)")
    # mambo.fly_direct(roll=25, pitch=0, yaw=50, vertical_movement=0, duration=3)

    print("landing")
    mambo.safe_land(5)
    mambo.smart_sleep(5)

    print("disconnect")
    mambo.disconnect()
