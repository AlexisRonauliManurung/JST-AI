print(">>>>> PERCEPTRON UNTUK GERBANG LOGIKA X-NOR <<<<<\n\n")

ambang_batas = 0.2

def aktivasi(keluaran, ambang_batas):
    if (keluaran > ambang_batas):
        return 1
    elif (keluaran <= ambang_batas and keluaran >= (ambang_batas * -1)):
        return 0
    else:
        return -1

def perbarui_bobot(x_i1, x_i2, w_1, w_2, t):
    a = 1
    w1 = w_1 + a * t * x_i1
    w2 = w_2 + a * t * x_i2
    return w1, w2

def perbarui_bias(b, t):
    a = 1
    b = b + a * t
    return b

x1 = [1, 1, 0, 0]
x2 = [1, 0, 1, 0]
keluaran_aktual = [1, -1, -1, 1]
bobot = [0, 0]

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

    for j in range(4):
        y_in = bias + bobot[0] * x1[j] + bobot[1] * x2[j]
        y_out = aktivasi(y_in, ambang_batas)
        k = k + 1
        print("Latihan - " + str(k))
        print("\nMasukan 1  = " + str(x1[j]))
        print("Masukan 2  = " + str(x2[j]))
        print("\nBias = " + str(bias))
        print("Bobot 1 = " + str(bobot[0]))
        print("Bobot 2 = " + str(bobot[1]))
        print("\nJumlah = " + str(y_in))
        print("\nKeluaran Aktual : " + str(keluaran_aktual[j]) + "\nKeluaran Prediksi: " + str(y_out))

        if (y_out != keluaran_aktual[j]):
            print(" \nMemperbarui Bobot")
            bias = perbarui_bias(bias, keluaran_aktual[j])
            bobot[0], bobot[1] = perbarui_bobot(x1[j], x2[j], bobot[0], bobot[1], keluaran_aktual[j])
            print("Bobot Diperbarui: " + str(bobot[0]) + ", " + str(bobot[1]))
            print("\nBobot Diperbarui, Melatih Ulang: ")
            print("==================================================")
            e = e + 1

print("\n\nSTATUS PROSES PELATIHAN: HOREE, SELESAI! ^^\n")
print("===================================================")
print("HASIL AKHIR")
print("===================================================")
print("\nEpoch = " + str(epoch))
print("Latihan = " + str(k))
print("Bias = " + str(bias))
print("Bobot 1 = " + str(bobot[0]))
print("Bobot 2 = " + str(bobot[1]))