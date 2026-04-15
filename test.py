from earningscall import get_company

company = get_company("aapl")  # Lookup Apple, Inc by its ticker symbol, "AAPL"

transcript = company.get_transcript(year=2021, quarter=3, level=2)

speaker = transcript.speakers[1]  # Get second speaker
speaker_label = speaker.speaker_info.name
text = speaker.text
print("Speaker:")
print(f"  Name: {speaker.speaker_info.name}")
print(f"  Title: {speaker.speaker_info.title}")
print()
#print(f"Text: {text}")