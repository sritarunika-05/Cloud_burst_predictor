from twilio.rest import Client
import keys
    
client=Client(keys.account_sid, keys.auth_token)

for i in keys.target_number:
    message=client.messages.create(
            body="THERE IS A CLOUD BURST IN YOUR LOCALITY! BE AWARE",
            from_=keys.twilio_number,
            to=i
        )

print(message.body)