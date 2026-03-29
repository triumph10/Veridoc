from src.ingestion.loader import load_pdf
pages = load_pdf("C:\\Users\\Siddhesh Patil\\OneDrive\\Desktop\\Projects\\veridoc\\Experiment 4 (2).pdf")

print(len(pages))
print(pages[0])