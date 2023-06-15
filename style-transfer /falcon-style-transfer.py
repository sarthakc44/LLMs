{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyOrLuOd41q7jIDJcByz4oAP"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "## Falcon-7b-instruct"
      ],
      "metadata": {
        "id": "MjtOHx0S8441"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install transformers einops accelerate"
      ],
      "metadata": {
        "id": "b81uL4cyq_8v"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import locale\n",
        "print(locale.getpreferredencoding())\n",
        "\n",
        "def getpreferredencoding(do_setlocale = True):\n",
        "    return \"UTF-8\"\n",
        "    # needed for gradio\n",
        "locale.getpreferredencoding = getpreferredencoding\n",
        "\n",
        "print(locale.getpreferredencoding())"
      ],
      "metadata": {
        "id": "I6Caqf2PJtwu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install gradio"
      ],
      "metadata": {
        "id": "EjRDGCNtJull"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        "import transformers, torch, gradio as gr"
      ],
      "metadata": {
        "id": "sp0l02F3rCEO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "model = \"tiiuae/falcon-7b-instruct\"\n",
        "\n",
        "tokenizer = AutoTokenizer.from_pretrained(model)\n",
        "pipeline = transformers.pipeline(\n",
        "    \"text-generation\",\n",
        "    model=model,\n",
        "    tokenizer=tokenizer,\n",
        "    torch_dtype=torch.bfloat16,\n",
        "    trust_remote_code=True,\n",
        "    device_map=\"auto\",\n",
        ")\n",
        "\n"
      ],
      "metadata": {
        "id": "uL7xW2PdeMEn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prompt = 'Paraphrase the following sentence delimited by curly brackets into'\n",
        "style = ' exaggerated victorian english: '\n",
        "input_text = '{' + 'Almost lunchtime. Time to eat!' + '}'\n",
        "\n",
        "text_prompt = prompt + style + input_text\n",
        "\n",
        "def llm(input_text, style):\n",
        "  prompt = 'Paraphrase and change the style of the following sentence delimited by curly brackets into an exaggerated '\n",
        "  style = style + ' accent: '\n",
        "\n",
        "  text_prompt = prompt + style + '{' + input_text + '}'\n",
        "\n",
        "  sequences = pipeline(\n",
        "      text_prompt,\n",
        "      max_length=256,\n",
        "      do_sample=True,\n",
        "      num_return_sequences=1,\n",
        "      eos_token_id=tokenizer.eos_token_id,\n",
        "      return_full_text=False\n",
        "  )\n",
        "\n",
        "  output_text = ''\n",
        "  for seq in sequences:\n",
        "      output_text = output_text + seq['generated_text']\n",
        "\n",
        "  return output_text\n",
        "\n",
        "#for seq in sequences:\n",
        " #   print(f\"Result: {seq['generated_text']}\")"
      ],
      "metadata": {
        "id": "i_wSlwGqopqT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "title = \"Change your Speaking Style!\"\n",
        "description = \"\"\"\n",
        "Write something, select an accent, and change the style of your text in seconds!\n",
        "\n",
        "Didn't like the response? Just click on submit again!\n",
        "\"\"\"\n",
        "\n",
        "article = \"This demo uses the [Falcon-7b-Instruct Model](https://huggingface.co/tiiuae/falcon-7b-instruct) and is purely for recreational purposes. View the source code on the [github repo.](https://github.com/sarthakc44/LLMs/tree/main/style-transfer)\"\n",
        "\n",
        "\n",
        "textbox = gr.Textbox(label=\"Type a few sentences below:\", placeholder=\"Almost lunchtime. Time to eat!\", lines=3)\n",
        "radio = gr.Radio([\"Crazy Pirate\", \"Formal Victorian\", \"Hillbilly Southern\", \"Casual Talkative\", \"Flowery Poetic\",], label=\"Choose your accent!\")\n",
        "\n",
        "demo = gr.Interface(\n",
        "  fn=llm,\n",
        "  inputs=[textbox,\n",
        "          radio,],\n",
        "  outputs=\"text\",\n",
        "  title=title,\n",
        "  description=description,\n",
        "  article=article,\n",
        ").launch()\n"
      ],
      "metadata": {
        "id": "QBDeMBB_rkIU",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 616
        },
        "outputId": "80ce27dc-20c2-4040-ca06-c821732b5341"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Colab notebook detected. To show errors in colab notebook, set debug=True in launch()\n",
            "Note: opening Chrome Inspector may crash demo inside Colab notebooks.\n",
            "\n",
            "To create a public link, set `share=True` in `launch()`.\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "(async (port, path, width, height, cache, element) => {\n",
              "                        if (!google.colab.kernel.accessAllowed && !cache) {\n",
              "                            return;\n",
              "                        }\n",
              "                        element.appendChild(document.createTextNode(''));\n",
              "                        const url = await google.colab.kernel.proxyPort(port, {cache});\n",
              "\n",
              "                        const external_link = document.createElement('div');\n",
              "                        external_link.innerHTML = `\n",
              "                            <div style=\"font-family: monospace; margin-bottom: 0.5rem\">\n",
              "                                Running on <a href=${new URL(path, url).toString()} target=\"_blank\">\n",
              "                                    https://localhost:${port}${path}\n",
              "                                </a>\n",
              "                            </div>\n",
              "                        `;\n",
              "                        element.appendChild(external_link);\n",
              "\n",
              "                        const iframe = document.createElement('iframe');\n",
              "                        iframe.src = new URL(path, url).toString();\n",
              "                        iframe.height = height;\n",
              "                        iframe.allow = \"autoplay; camera; microphone; clipboard-read; clipboard-write;\"\n",
              "                        iframe.width = width;\n",
              "                        iframe.style.border = 0;\n",
              "                        element.appendChild(iframe);\n",
              "                    })(7872, \"/\", \"100%\", 500, false, window.element)"
            ]
          },
          "metadata": {}
        }
      ]
    }
  ]
}