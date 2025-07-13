# Modal Ace-Step
This is a simple implementation of the Ace-Step audio generation ML model using modal.com. Modal.com is a serverless GPU provider.

The implementation makes use of model weight caching on a modal volume, as well as an attempt to use memory snapshotting.

It's used by www.text-to-sample.com

## How to use
1. Create an account on modal
2. install modal 
`pip install modal`
3. Deploy the modal_app.py app with:
`modal deploy modal_app.py`
4. Done!

This will create two web endpoints. One for audio-to-audio, and one for text-to-audio.

I have had a hard time getting any good results with audio-to-audio, but text-to-audio works great.

You can specify lyrics, or leave it empty for instrumental only.

The raw audio bytes will be returned by the endpoint.

You can also just run the modal app with:
`modal run modal_app.py`
