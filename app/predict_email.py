import pandas as pd
import json
import re
from bs4 import BeautifulSoup
import email
import urllib
import base64
import string
import quopri
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Prediction:
    def __init__(self):
        vocab = joblib.load("./app/data/vocab.sav")
        self.model = joblib.load("./app/data/model.sav")
        self.tfidf = TfidfVectorizer(vocabulary=vocab)
        
    @staticmethod
    def parse_email(email_raw):
        email_payload = email_raw.get_payload()

        email_body = ""

        if isinstance(email_payload, list):
            for part in email_payload:
                email_body += str(Prediction.parse_email(part))

            return email_body
        else:
            if "Content-Type" in email_raw:
                try: # To debug why the exception "'Header' object has no attribute 'lower' occures"
                    if "text/html" in email_raw["Content-Type"].lower() or "text/plain" in email_raw["Content-Type"].lower(): # only parse content of type "text/html" and "text/plain"
                        if "Content-Transfer-Encoding" in email_raw:
                            if email_raw["Content-Transfer-Encoding"].lower() == "base64": # check if its base64 encoded
                                try:
                                    return str(base64.b64decode(email_payload))
                                except:       # if the decoding did not work
                                    return "" # just return an empty string
                            elif email_raw["Content-Transfer-Encoding"].lower() == "quoted-printable":
                                try:
                                    email_payload = ''.join(filter(lambda x: x in string.printable, email_payload))
                                    return str(quopri.decodestring(email_payload))
                                except:       # if the decoding did not work
                                    return "" # just return an empty string
                            else:
                                return email_payload
                        else:
                            return email_payload
                except Exception as e:
                        print(f"Failed email parsing:\nSubject: {email_raw['subject']} | From: {email_raw['From']} | email_raw[\"Content-Type\"]='{email_raw['Content-Type']}'\n{e}\n\n")
            elif email_raw.get_default_type() == "text/plain":
                # If the there is no "Content-Type" and the default type is "text/plain"
                #print(email_payload)
                return email_payload
            else:
                return ""

    @staticmethod
    def parse_dataset(dataset):
        rows = []
        parser = email.parser.BytesParser()
        re_email = re.compile("[\w.-]+@[\w.-]+.[\w.-]+", re.UNICODE)

        for mail in dataset:
            with open(mail, "rb") as f:

                email_raw = parser.parse(f)

                #print(email_raw['subject'])
                subject_mail = email_raw['subject']
                #print(email_raw['From'])
                from_mail = email_raw['From']
                if from_mail != None:
                    try:
                        from_mail = re_email.search(str(from_mail)).group()
                    except AttributeError:
                        print("Could not parse email 'from' header")
                        from_mail = None

                # very important, if not none --> BAAAAAMM Spam
                #print(email_raw['X-Authentication-Warning'])
                auth_error_mail = email_raw['X-Authentication-Warning']
                if auth_error_mail == None:
                    auth_error_mail = 0

                email_payload = Prediction.parse_email(email_raw)# email_raw.get_payload()
                if email_payload == None:
                    print("Could not parse email body")

                if email_payload == None or len(email_payload) == 0:
                    email_payload = "0"

                #urls = re.findall(r'(https?://[^\s"\']+)', email_payload)
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[-$_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_payload)
                domains = []
                if len(urls) == 0:
                    urls = 0
                else:
                    # parsed domains (not /....)
                    #domains = [ urllib.parse.urlparse(url).netloc for url in urls ]
                    for url in urls:
                        try:
                            domains.append(re.sub(":\d+", "", urllib.parse.urlparse(url).netloc))
                            #domains.append(urllib.parse.urlparse(url).netloc)
                        except:
                            print("Could not parse domain") 

                if len(domains) == 0:
                    domains = 0

                if BeautifulSoup(email_payload.encode("utf-8"), "html.parser").find():
                    #print("HTML")
                    cleantext = BeautifulSoup(email_payload, "html.parser").text
                else:
                    cleantext = email_payload

                clean_data = cleantext
                clean_data.replace("\n", " ")

                #print(cleantext)
                #print(email_raw.keys())

                # spam/ham, ascii-mail, subject, from, to, auth-error, urls
                rows.append([cleantext, subject_mail, from_mail, auth_error_mail, urls, domains])
                #rows.append([1, cleantext, urls])

        return rows

    @staticmethod
    def clean_text(text):
        # remove replies and forwards
        start_reply = re.search(r"\nOn .* wrote:", text)
        if start_reply != None:
            cleared_text = text[:start_reply.start()]
        else:
            cleared_text = text

        # remove \n or \r or \\n or \\r
        cleared_text = cleared_text.replace('\n', ' ').replace('\r', ' ').replace('\\n', ' ').replace('\\r', ' ')

        # remove URLs
        cleared_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[-$_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', cleared_text)

        # remove email addrs
        re_email = re.compile("[\w.-]+@[\w.-]+.[\w.-]+", re.UNICODE)
        cleared_text = re_email.sub(' ', cleared_text)

        # replace non-alpha chars with space
        cleared_text = re.sub('[^a-zA-Z]', ' ', cleared_text)

        # convert everything to lowercase
        cleared_text = cleared_text.lower()

        cleared_text = cleared_text.split()
        cleared_text = ' '.join(cleared_text)

        return cleared_text

    @staticmethod
    def remove_stopwords(text):
        stop_words = None
        try:
            stop_words = set(stopwords.words("english"))
        except:
            download('stopwords')
            stop_words = set(stopwords.words("english"))
        filtered_text = [word for word in text if word not in stop_words]
        return filtered_text

    @staticmethod
    def lemmatize_word(text):
        lemmatizer = WordNetLemmatizer()
        lemmas = None
        try:    
            lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in text]
        except:        
            download('wordnet')
            lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in text]
        return lemmas

    def predict_emails(self, emails):
        rows = Prediction.parse_dataset(emails)
        df = pd.DataFrame(rows, columns = ['raw_data', 'subject', 'from', 'auth_error', 'urls', 'domains'])

        df["data"] = df["raw_data"].apply(Prediction.clean_text)

        try:                                  
            df["token_text"] = df.apply(lambda row: word_tokenize(str(row["data"])), axis=1)
        except:
            download('punkt')
            df["token_text"] = df.apply(lambda row: word_tokenize(str(row["data"])), axis=1)
            
        df["stop_text"] = df["token_text"].apply(Prediction.remove_stopwords)                                 
        df["clean_text"] = df["stop_text"].apply(Prediction.lemmatize_word)                                  
        df["chars"] = df["raw_data"].apply(len)
        df['tokens'] = df['token_text'].str.len()

        corpus = []
        for text in df["clean_text"]:
            msg = ' '.join([row for row in text])
            corpus.append(msg)

        X = self.tfidf.fit_transform(corpus).toarray()

        X = np.append(X, df[["tokens", "chars"]].to_numpy(), axis=1)


        spam = self.model.predict(X)
        probability = self.model.predict_proba(X)

        return (spam, probability)

def example():    
    emails = ['.\\datasets\\custom\\test5.eml', '.\\datasets\\trec07p\\spam\\inmail.100', '.\\datasets\\trec07p\\ham\\inmail.10845']
    
    p = Prediction()
    spam, probability = p.predict_emails(emails)
    print(f"Spam={spam} | Probability={probability}")