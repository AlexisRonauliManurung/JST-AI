print(">>>>> PERCEPTRON UNTUK GERBANG LOGIKA NOT <<<<<\n\n")

ambang_batas = 0.2

def aktivasi(keluaran, ambang_batas):
    if (keluaran > ambang_batas):
        return 1
    elif (keluaran <= ambang_batas and keluaran >= (ambang_batas * -1)):
        return 0
    else:
        return -1

def perbarui_bobot(x, w, t):
    a = 1
    w = w + a*t*x
    return w

def perbarui_bias(b, t):
    a = 1
    b = b + a*t
    return b

x1 = [0, 1]
keluaran_aktual = [1, -1]
bobot = [0]

bias = 0

j = 0
k = 0
e = 1

epoch = 0

#while key = 0:
while e > 0:

    e = 0
    epoch = epoch + 1

    print("\n===========================")
    print("         EPOCH - " + str(epoch))
    print("===========================\n")

    for j in range(2):
        y_in = bias + bobot[0] * x1[j]
        y_out = aktivasi(y_in, ambang_batas)
        k = k + 1
        print("Latihan - " + str(k))
        print("\nMasukan = " + str(x1[j]))
        print("\nBias = " + str(bias))
        print("Bobot = " + str(bobot[0]))
        print("\nJumlah = " + str(y_in))
        print("\nKeluaran Aktual : " + str(keluaran_aktual[j]) + "\nKeluaran Prediksi: " + str(y_out))

        if (y_out != keluaran_aktual[j]):
            print(" \nMemperbarui Bobot")
            bias = perbarui_bias(bias, keluaran_aktual[j])
            bobot[0] = perbarui_bobot(x1[j], bobot[0], keluaran_aktual[j])
            print("Bobot Diperbarui: " + str(bobot[0]))
            print("\nBobot Diperbarui, Melatih Ulang: ")
            print("==================================================")
            e = e + 1

print("\nSTATUS PROSES PELATIHAN: HOREE, SELESAI! ^^\n")
print("===================================================")
print("HASIL AKHIR")
print("===================================================")
print("\nEpoch = " + str(epoch))
print("Latihan = " + str(k))
print("Bias = " + str(bias))
print("Bobot = " + str(bobot[0]))