{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNSYswVd+GDaIRjeNNiLIxD",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/StevenDreamer1/DATA-MAINING/blob/main/exp2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.datasets import imdb\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Dropout\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences"
      ],
      "metadata": {
        "id": "HfS3Evq9tmQ5"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "(x_train, y_train),(x_test, y_test)=imdb.load_data(num_words=30000)"
      ],
      "metadata": {
        "id": "MKobmV4Qt3HX"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "maxlen = 400\n",
        "x_train = pad_sequences(x_train, maxlen=maxlen)\n",
        "x_test = pad_sequences(x_test, maxlen=maxlen)"
      ],
      "metadata": {
        "id": "Gd_tsAM-uTqh"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = Sequential()\n",
        "model.add(Dense(256, activation='relu', input_shape=(maxlen,)))\n",
        "model.add(Dropout(0.5))\n",
        "model.add(Dense(64, activation='relu'))\n",
        "model.add(Dropout(0.5))\n",
        "model.add(Dense(1, activation='sigmoid'))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KJNLqvZfugB3",
        "outputId": "8267598f-724b-472e-a938-9df674e157d1"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
        "model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=12, batch_size=32)\n",
        "scores=model.evaluate(x_test, y_test, verbose=0)\n",
        "print(\"Accuracy: %.2f%%\" % (scores[1]*100))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Oma77E9pvUvS",
        "outputId": "da0fdca7-801a-4990-9b2d-0d2050e3d414"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 4ms/step - accuracy: 0.4958 - loss: 0.6932 - val_accuracy: 0.5000 - val_loss: 0.6932\n",
            "Epoch 2/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 4ms/step - accuracy: 0.5024 - loss: 0.6937 - val_accuracy: 0.5000 - val_loss: 0.6932\n",
            "Epoch 3/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 5ms/step - accuracy: 0.4949 - loss: 0.7014 - val_accuracy: 0.5000 - val_loss: 0.6931\n",
            "Epoch 4/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 4ms/step - accuracy: 0.5025 - loss: 0.7309 - val_accuracy: 0.5000 - val_loss: 0.6931\n",
            "Epoch 5/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 4ms/step - accuracy: 0.5093 - loss: 0.6934 - val_accuracy: 0.5000 - val_loss: 0.6932\n",
            "Epoch 6/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 8ms/step - accuracy: 0.5025 - loss: 0.6943 - val_accuracy: 0.5000 - val_loss: 0.6931\n",
            "Epoch 7/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m9s\u001b[0m 6ms/step - accuracy: 0.4951 - loss: 0.7379 - val_accuracy: 0.5000 - val_loss: 0.6931\n",
            "Epoch 8/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 4ms/step - accuracy: 0.4930 - loss: 0.6946 - val_accuracy: 0.5000 - val_loss: 0.6932\n",
            "Epoch 9/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 4ms/step - accuracy: 0.4925 - loss: 0.6932 - val_accuracy: 0.5000 - val_loss: 0.6931\n",
            "Epoch 10/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 7ms/step - accuracy: 0.4930 - loss: 0.6976 - val_accuracy: 0.5000 - val_loss: 0.6932\n",
            "Epoch 11/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 4ms/step - accuracy: 0.5039 - loss: 0.7004 - val_accuracy: 0.5000 - val_loss: 0.6931\n",
            "Epoch 12/12\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 5ms/step - accuracy: 0.5024 - loss: 0.6950 - val_accuracy: 0.5000 - val_loss: 0.6932\n",
            "Accuracy: 50.00%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import classification_report\n",
        "y_pred = (model.predict(x_test) > 0.5).astype(\"int32\")\n",
        "print(classification_report(y_test, y_pred, target_names=[\"Negative\", \"Positive\"]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "myETDhENweJh",
        "outputId": "85dfdadc-bfd6-430e-fd1a-db057275d54b"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 1ms/step\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "    Negative       0.50      0.98      0.66     12500\n",
            "    Positive       0.48      0.02      0.04     12500\n",
            "\n",
            "    accuracy                           0.50     25000\n",
            "   macro avg       0.49      0.50      0.35     25000\n",
            "weighted avg       0.49      0.50      0.35     25000\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[\"Negative\", \"Positive\"],\n",
        "yticklabels=[\"Negative\", \"Positive\"])\n",
        "plt.xlabel(\"Predicted\")\n",
        "plt.ylabel(\"Actual\")\n",
        "plt.title(\"Confusion Matrix\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 564
        },
        "id": "NISyF2siwtUp",
        "outputId": "f25b9db6-3f18-4c4f-d4e4-dbab0b099fbc"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x600 with 2 Axes>"
            ],
            "image/png": "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r1497d69WwkJCSpfvrxNvwoVKsjX11fx8fEqXry4vLy8VKdOHYWGhuYrrqSkJH388cd65513rNstff7553rsscc0fPhwxcbG5ms8ALhbqGgCDu7JJ5/Url279Mwzz+i///2voqOjNXToUB09elTjxo1TXFycte9nn32mkSNHasuWLerXr5+SkpI0bNgwff311wUaU4kSJbRgwQJ5enrq9ddf18yZMxUTE6PWrVvnir1s2bL6z3/+o+joaH300Udq1KiRkpKS5OPjc9PxIyIitGLFCpUoUUJvv/22PvzwQ9WtW1cbNmzId5JmhDfeeEMDBw7UypUr1bdvX23btk1Lly5VmTJlbPoVLVpUM2fOVJEiRdSzZ089//zzWrt2bb6udfHiRXXt2lU1atTQm2++aW1v2LCh+vbtq3HjxunHH38skPsCgIJmsuRntTwAAABwi6hoAgAAwBAkmgAAADAEiSYAAAAMQaIJAAAAQ5BoAgAAwBAkmgAAADAEiSYAAAAMUSh/GcijRm97hwDAIOe3TLF3CAAM4m7HrMTI3OHyduf97xYVTQAAABiiUFY0AQAA8sVE7c0IJJoAAAAmk70jKJRI3wEAAGAIEk0AAACTi3GvfFq3bp1at26t4OBgmUwmLVy40HosMzNTQ4YMUbVq1eTl5aXg4GB17txZJ06csBnj3Llz6tSpk7y9veXr66tu3bopLS3Nps+uXbvUsGFDubu7q0yZMoqNjc0VyzfffKPKlSvL3d1d1apV07Jly/J1LySaAAAADuTSpUt65JFH9NFHH+U6lp6erm3btmn48OHatm2bvv32Wx04cEBPPvmkTb9OnTppz549SkxM1JIlS7Ru3Tr16NHDejw1NVVNmzZVSEiItm7dqg8++EAjRozQtGnTrH02btyo559/Xt26ddP27dvVpk0btWnTRr/88sst34vJYrFYbuMzcGhsbwQUXmxvBBRedt3e6NEBho19YX2MMjIybNrMZrPMZvM/nmsymbRgwQK1adPmpn22bNmif/3rXzp27JjKli2rffv2qWrVqtqyZYtq164tSVqxYoVatGihP/74Q8HBwZo6darefPNNJScny83NTZI0dOhQLVy4UPv375ckPffcc7p06ZKWLFlivVbdunVVvXp1xcfH39K9U9EEAAAwUExMjHx8fGxeMTExBTZ+SkqKTCaTfH19JUmb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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}