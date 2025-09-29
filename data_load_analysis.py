import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

#Ucitavanje excel fajla radi daljeg obradjivanja,u slucaju greske vraca none
def load_excel_data(file_path):

    try:
        df = pd.read_csv(file_path)
        print(f"Uspesno ste ucitali fajl: {file_path}")
        return df
    except Exception as e:
        print(f"Greska pri ucitanvanju fajla: {e}")
        return None

#Neophodna provera koju sam zbog svoje orijentacije odradio,da li kolona ima Nan file
#predpostavio sam da za neke karakteristike kao sto su model ili regija da nema smisla ali za svaki slucaj
#Na kraju treniranje modela su odradjena samo bez parametra regija,u slucaju izbacivanja
#regije i sales volume proizvodi katastrofalnu gresku

#Proveerava kolicinu Nullova u datasetu
def null_statistic(df):
    print("Provera da li se NULL pojavio u datoj koloni")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"Kolona ima NULL vrednost {col}: {null_count}")
        else:
            print(f"Kolona {col} nema Null vrednost")

#Najvaznija funkcija za ceo projekat,obradjuje podatatke i trazi anomalije
#Implementirano na kraju kroz posmatranje medijana,pokusao sam primeniti preko IQR i Z_scorea,
#Zadovoljavajuce podatke sam na kraju dobio kroz pretragu sa medijanom,
#Nesiguran da li bi radilo preko Z_scorea jer mi je bilo tesko odrediti threshold za
#koji ne bih dobijao mnogo anomalija(40-60% svih podataka)
def anomaly_detection_num_values(df,tolerance = 0.85,z_threshold = 4.0,strategy_im = 'mean',variation_const = 0.05):
    #Ucitavanje i fokusirano na float i int podatke,str kolone podatak obradjivane dalje enkodiranjem
    df_copy = df.copy()
    num_column = df_copy.select_dtypes(include=['float64', 'int64']).columns

    anomaly_report = {}
        #Pokusaj preko srednje vrednosti
        #mean = df_copy[col].mean()
        #lower_limit = mean * (1 - tolerance)
        #upper_limit = mean * (1 + tolerance)

        # Pokusaj Z score implementacije
        # std = df_copy[col].std()
        # z_scores = (df_copy[col] - mean) / std
        # anomaly_mask = z_scores.abs() < z_threshold

        #Pokusaj preko IQR scorea
        #q1 = df[col].quantile(0.25)
        #q3 = df[col].quantile(0.75)
        #iqr = q3 - q1

        #lower_limit_q = q1 - 1.5 * iqr
        #upper_limit_q = q3 + 1.5 * iqr

    for col in num_column:
        #Implementacije obrade preko medijana
        median = df_copy[col].median()
        lower_limit_median = median * (1 - tolerance)
        upper_limit_median = median * (1 + tolerance)

        #Vraca masku koja govori
        # True - ne nalazi se izmedju limita
        #False - nalazi se izmedju linija
        #Originalno je obrnuto ali iskoristeno ~
        anomaly_mask = ~df_copy[col].between(lower_limit_median, upper_limit_median)
        num_anomalies = anomaly_mask.sum()

        #Ako postoje podaci koji su anomalije obradjujemo ih
        if anomaly_mask.any():
            #Dtype nad kojim radimo da bi znali kasnije za zaokruzivanje,ako ne zaokruzim iskako mi error
            col_dtype = df_copy[col].dtype

            #Pokusao da dodam varijacioni faktor da model ne bi bio previse naviknut
            #na jednom citane dataset,nisam siguran koliko je na kraju uticao na samu
            #Obuku modela
            variation_factor = np.random.uniform(1 - variation_const, 1 + variation_const)
            #mean_variated = mean * variation_factor
            median_variated = median * variation_factor

            if col_dtype == 'int64':
                #mean_new = int(round(mean_variated))
                median_new = int(round(median_variated))
            else:
                #mean_new = mean_variated
                median_new = median_variated

            #Koriscena provera jer nisam mogao da obuzdam obradu
            #Previse anomalija pa sam proveravao da li je npr 1.8 motor anomalija
            #ili da ga gledam kao validan podatak
            #if col == 'Engine_Size_L':
                #below_2 = df_copy[col] < 1.8
                #above_4_4 = df_copy[col] > 4.6
                #print(f"{below_2.sum()} broj auta ispod 2.0")
                #print(f"{above_4_4.sum()} broj auta iznad 4.4")

            #Svaki podatak koji je detektovan kao anomalija je uzet da bude
            df_copy.loc[anomaly_mask,col] = median_new
            anomaly_report[col] = num_anomalies

        #U mom datasetu se nisu pojavljivali Nan vrednosti
        #U svakom slucaju ubacio sam imputaciju koja je namenjena da kad se obrade anomalija
        #Ubacujemo medijan u sve nan vrednosti,mislim da ovo malo stvara veliki bias oko medijan vrednosti
        #no model je svakako prikazivao zadovoljavajuce rezultate
        if df_copy[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy=strategy_im)
            df_copy[[col]] = imputer.fit_transform(df_copy[[col]])

    #Ostavio na kraju kao prikaz od testiranja
    for col,count in anomaly_report.items():
        print(f"{col}: {count}")
    #print(f"Detektovano anomalija")
    #Vracamo modifikovan dataframe
    return df_copy

#Zamisljeno da detektuje anomalije medju string podacima ali u krajnoj implementaciji nije bio koristen
#def anomaly_detection_str_values(df, min_frq = 10):
    #df_copy = df.copy()
    #str_columns = df_copy.select_dtypes(include='object').columns

    #for col in str_columns:
        #value_count = df_copy[col].value_counts()
        #allowed_values = value_count[value_count>=min_frq].index.tolist()

        #anomaly_mask = ~df_copy[col].isin(allowed_values)
        #df_copy[f"{col}_is_anomaly"] = anomaly_mask.astype(int)
        #print(f"Kolona '{col}': {anomaly_mask.sum()} detektovanih anomalija")

    #return df_copy
#Jos jedna funkcija koja enkodira sve nase string kolone sto
#omogucava nasem modelu da se trenira nad njima
#Nemoguce je nad stringovima pa zato radimo ovo
def onehot_encode(df, target_col = 'Sales_Classification'):
    df_copy = df.copy()

    cat_cols = df_copy.select_dtypes(include='object').columns
    encode_cols = [col for col in cat_cols if col != target_col]
    df_encoded = pd.get_dummies(df_copy,columns = encode_cols,drop_first = True)

    return df_encoded
#Odvojeno enkodiranje za target kolonu
def encode_target_label(df,target_col ='Sales_Classification'):
    df_copy = df.copy()

    le = LabelEncoder()
    df_copy[target_col] = le.fit_transform(df_copy[target_col])

    return df_copy,le




