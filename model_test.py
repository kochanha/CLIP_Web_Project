from model import clip_class

if __name__ == "__main__":
    a = 0

    for i in range(100):
        gt, rfile = clip_class.get_ramdom_path()
        predict, rfile = clip_class.clip_predict(rfile)

        print('stage: ', i)
        print("Ground Truth :", gt)
        print("클래스 예측 : ", predict)

        if gt == predict:
            a+=1
    print("ViT-B/32 Accurcy:", a/100)
