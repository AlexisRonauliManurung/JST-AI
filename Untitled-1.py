print(">>>>> PERCEPTRON UNTUK GERBANG LOGIKA AND <<<<<\n\n")

ambang = 0.2

def aktivasi(keluar, ambang):
    if (keluar > ambang):
        return 1
    elif (keluar <= ambang and keluar >= (ambang * -1)):
        return 0
    else:
        return -1

def ubah_bobot(x_i1, x_i2, b_1, b_2, t):
    a = 1
    bobot1 = b_1 + a * t * x_i1
    bobot2 = b_2 + a * t * x_i2
    return bobot1, bobot2

def ubah_bias(b, t):
    a = 1
    b = b + a * t
    return b

x1 = [1, 1, 0, 0]
x2 = [1, 0, 1, 0]
keluar_aktual = [1, -1, -1, -1]
bobot = [0, 0]

bias = 0

j = 0
k = 0
e = 1

epoch = 0

# while key = 0:
while e > 0:

    e = 0
    epoch = epoch + 1

    print("\n=======================================")
    print("         EPOCH -       " + str(epoch))
    print("=======================================\n")

    for j in range(4):
        y_in = bias + bobot[0] * x1[j] + bobot[1] * x2[j]
        y_out = aktivasi(y_in, ambang)
        k = k + 1
        print("Pelatihan yang ke-" + str(k))
        print("\nInput 1  = " + str(x1[j]))
        print("Input 2  = " + str(x2[j]))
        print("\nBias = " + str(bias))
        print("Bobot 1 = " + str(bobot[0]))
        print("Bobot 2 = " + str(bobot[1]))
        print("\nPenjumlahan = " + str(y_in))
        print("\nOutput Aktual : " + str(keluar_aktual[j]) + "\nOutput Prediksi: " + str(y_out))

        if (y_out != keluar_aktual[j]):
            print(" \nMemperbarui Bobot")
            bias = ubah_bias(bias, keluar_aktual[j])
            bobot[0], bobot[1] = ubah_bobot(x1[j], x2[j], bobot[0], bobot[1], keluar_aktual[j])
            print("Bobot Diperbarui: " + str(bobot[0]) + ", " + str(bobot[1]))
            print("\nBobot Diperbarui Melatih Kembali: ")
            print("==================================================")
            e = e + 1

print("\n\nSTATUS PROSES PELATIHAN: HORREE, SELESAI DAN TUNTAS! ^^")