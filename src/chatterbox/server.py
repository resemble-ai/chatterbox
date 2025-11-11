    import socket

    test_request = "Hi, I'm Delta's AI assistant! How can I help you today?"

    sample_rate = 24000
    frame_size = 210
    frame_duration = frame_size / sample_rate
    dtype = np.float32

    HOST = "0.0.0.0"
    PORT = 9000
    server.bind((HOST, PORT))
    server.listen(1)
    print("waiting for connection...")
    client, addr = server.accept()
    print("client connected to server!\n")

        with client:

        
    #     # initializing and starting chatterbox stream
    #     stream = ChatterboxStreamer(
    #         sample_rate = sample_rate,
    #         fade_duration = 0.02,
    #         dtype=dtype
    #     )
    #     stream.start()
        
    #     # making test request
    #     start_time = time.time()
    #     stream.make_request((test_request, start_time))
        

    #     client.sendall("Hello".encode())
    #     client.close()

    # streamer = ChatterboxStreamer(
    #     sample_rate = sample_rate,
    #     fade_duration = 0.02,
    #     dtype=dtype
    # )

    # # text = [
    # #     "Active-duty U.S. military personnel get special baggage allowances with Delta. When traveling on orders or for personal travel, you’ll receive baggage fee exceptions and extra checked bag benefits. These allowances apply to all branches, including the Marine Corps, Army, Air Force, Space Force, Navy, and Coast Guard. There may be some regional weight or embargo restrictions. Would you like me to text you a link with the full details for military baggage policies?", 
    # #     "Yes, there are specific restrictions for minors and unaccompanied minors traveling internationally with Delta Air Lines. For international travel, Delta requires that all passengers under the age of fifteen use the Unaccompanied Minor Service. This service provides supervision from boarding until the child is met at their destination."
    # # ]



    # audio = np.zeros(0, dtype=dtype)
    # start_time = time.time()
    # streamer.make_request((request, start_time))
    # terminate = False

    # while True:
    #     frame, request_finished = streamer.get_frame(frame_size)
    #     audio = np.concatenate([audio, frame])
    #     time.sleep(frame_duration)

    #     if request_finished: 
    #         print(f"Total generation time: {time.time() - start_time}")
    #         terminate = True
        
    #     if streamer.available_samples() == 0 and terminate:
    #         break
        
    
    # print(f"Total audio play time: {time.time() - start_time}")
    # sf.write("stream_snapshot.wav", audio, sample_rate)