# "isic_id","target","patient_id","age_approx","sex","anatom_site_general","clin_size_long_diam_mm","image_type","tbp_tile_type","tbp_lv_A","tbp_lv_Aext","tbp_lv_B","tbp_lv_Bext","tbp_lv_C","tbp_lv_Cext","tbp_lv_H","tbp_lv_Hext","tbp_lv_L","tbp_lv_Lext","tbp_lv_areaMM2","tbp_lv_area_perim_ratio","tbp_lv_color_std_mean","tbp_lv_deltaA","tbp_lv_deltaB","tbp_lv_deltaL","tbp_lv_deltaLB","tbp_lv_deltaLBnorm","tbp_lv_eccentricity","tbp_lv_location","tbp_lv_location_simple","tbp_lv_minorAxisMM","tbp_lv_nevi_confidence","tbp_lv_norm_border","tbp_lv_norm_color","tbp_lv_perimeterMM","tbp_lv_radial_color_std_max","tbp_lv_stdL","tbp_lv_stdLExt","tbp_lv_symm_2axis","tbp_lv_symm_2axis_angle","tbp_lv_x","tbp_lv_y","tbp_lv_z","attribution","copyright_license","lesion_id","iddx_full","iddx_1","iddx_2","iddx_3","iddx_4","iddx_5","mel_mitotic_index","mel_thick_mm","tbp_lv_dnn_lesion_confidence"
# "ISIC_0015670",0,"IP_1235828",60,"male","lower extremity",3.04,"TBP tile: close-up","3D: white",20.2444222995213,16.2619751791219,26.9224473862421,23.954773134576,33.6846375593731,28.9531171492014,53.0585445070501,55.8289238915569,54.3674478530884,62.0257012013125,3.15256145618315,27.4761698440053,0,3.98244712039942,2.96767425166613,-7.65825334822409,8.36056573901858,5.78430238218227,0.901301769055263,"Right Leg - Upper","Right Leg",1.54301591115455,0.0026285923013347,7.09135958738072,0,9.30700349273346,0,2.03619534985743,2.63778008066265,0.59047619047619,85,-182.703552246094,613.49365234375,-42.4279479980469,"Memorial Sloan Kettering Cancer Center","CC-BY","","Benign","Benign","","","","","",NA,97.5172817707062


import pandas as pd
d = pd.read_csv("data/train-metadata.csv", low_memory=False)
print("lignes, colonnes :", d.shape)
print("patients distincts :", d["patient_id"].nunique())
tailles = d.groupby("patient_id").size()
print("lesions par patient :", tailles.min(), int(tailles.median()), tailles.max())

SUSPECTES = ["iddx_full", "iddx_1", "mel_thick_mm", "mel_mitotic_index", "tbp_lv_dnn_lesion_confidence", "lesion_id"]
for c in SUSPECTES:
    rempli = d[c].notna().sum()
    malignes = d.loc[d[c].notna(), "target"].sum() # =est maligne si la valeur de la colonne target est 1
    print(f"{c:<22} rempli {rempli:>7} dont malignes {int(malignes):>4}")

na = d.isna().sum()
print(na[na > 0].sort_values(ascending=False)) # affiche les colonnes avec des valeurs manquantes et le nombre de valeurs manquantes pour chaque colonne, triées par ordre décroissant.
constantes = [c for c in d.columns if d[c].nunique(dropna=True) <= 1]
print("constantes :", constantes) 

#| Colonnes comportant au moins une valeur manquante, hors colonnes de fuite          | 
print("Colonnes comportant au moins une valeur manquante, hors colonnes de fuite :")
for c in d.columns:
    if c not in SUSPECTES and d[c].isna().sum() > 0:
        print(f" - {c:<22} : {d[c].isna().sum()} valeurs manquantes, soit {d[c].isna().mean():.2%} du total")

# | Pour chacune : taux de valeurs manquantes    
print("Pour chacune : taux de valeurs manquantes :")
for c in d.columns:
    if c not in SUSPECTES and d[c].isna().sum() > 0:
        print(f" - {c:<22} : {d[c].isna().mean():.2%} du total")

        
# | Colonnes constantes, c'est-à-dire à une seule modalité                             |                  |
print("Colonnes constantes, c'est-à-dire à une seule modalité :")
for c in constantes:
    print(f" - {c:<22} : {d[c].nunique(dropna=True)} modalité(s)")

# | Nombre de variables restantes après retrait des fuites, identifiants et constantes
print("Nombre de variables restantes après retrait des fuites, identifiants et constantes :")
remaining_columns = [c for c in d.columns if c not in SUSPECTES and c not in constantes]
print(f" - {len(remaining_columns)} variables restantes : {remaining_columns}")


# | Nombre de lésions<br>`target = 1` |                  |
print("Nombre de lésions avec target = 1 (malignes) :", d["target"].sum())

# | Nombre de lésions<br>`target = 0` |                  |
print("Nombre de lésions avec target = 0 (bénignes) :", (d["target"] == 0).sum())

# | Taux de positifs, en pourcentage  |                  |
print("Taux de positifs (malignes) :", d["target"].mean() * 100, "%")
