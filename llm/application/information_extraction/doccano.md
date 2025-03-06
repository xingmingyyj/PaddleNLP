# doccano

 **目录**

* [1. 安装](#安装)
* [2. 项目创建](#项目创建)
* [3. 数据上传](#数据上传)
* [4. 标签构建](#标签构建)
* [5. 任务标注](#任务标注)
* [6. 数据导出](#数据导出)
* [7. 数据转换](#数据转换)

<a name="安装"></a>

## 1. 安装

参考[doccano 官方文档](https://github.com/doccano/doccano) 完成 doccano 的安装与初始配置。

**以下标注示例用到的环境配置：**

- doccano 1.6.2

<a name="项目创建"></a>

## 2. 项目创建

PP-UIE 支持抽取类型的任务，根据实际需要创建一个新的项目：

#### 2.1 抽取式任务项目创建

创建项目时选择**序列标注**任务，并勾选**Allow overlapping entity**及**Use relation Labeling**。适配**命名实体识别、关系抽取、事件抽取**等任务。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167249142-44885510-51dc-4359-8054-9c89c9633700.png height=230 hspace='15'/>
</div>

<a name="数据上传"></a>

## 3. 数据上传

上传的文件为 txt 格式，每一行为一条待标注文本，示例:

```text
2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌
第十四届全运会在西安举办
```

上传数据类型**选择 TextLine**:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167247061-d5795c26-7a6f-4cdb-88ad-107a3cae5446.png height=300 hspace='15'/>
</div>

**NOTE**：doccano 支持`TextFile`、`TextLine`、`JSONL`和`CoNLL`四种数据上传格式，PP-UIE 定制训练中**统一使用 TextLine**这一文件格式，即上传的文件需要为 txt 格式，且在数据标注时，该文件的每一行待标注文本显示为一页内容。

<a name="标签构建"></a>

## 4. 标签构建

#### 4.1 构建抽取式任务标签

抽取式任务包含**Span**与**Relation**两种标签类型，Span 指**原文本中的目标信息片段**，如实体识别中某个类型的实体，事件抽取中的触发词和论元；Relation 指**原文本中 Span 之间的关系**，如关系抽取中两个实体（Subject&Object）之间的关系，事件抽取中论元和触发词之间的关系。

Span 类型标签构建示例:

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248034-afa3f637-65c5-4038-ada0-344ffbd776a2.png height=300 hspace='15'/>
</div>

Relation 类型标签构建示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248307-916c77f6-bf80-4d6b-aa71-30c719f68257.png height=260 hspace='16'/>
</div>


## 5. 任务标注

#### 5.1 命名实体识别

命名实体识别（Named Entity Recognition，简称 NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，**抽取的类别没有限制，用户可以自己定义**。

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248557-f1da3694-1063-465a-be9a-1bb811949530.png height=200 hspace='20'/>
</div>

示例中定义了`时间`、`选手`、`赛事名称`和`得分`四种 Span 类型标签。

```text
schema = [
    '时间',
    '选手',
    '赛事名称',
    '得分'
]
```

#### 5.2 关系抽取

关系抽取（Relation Extraction，简称 RE），是指从文本中识别实体并抽取实体之间的语义关系，即抽取三元组（实体一，关系类型，实体二）。

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248502-16a87902-3878-4432-b5b8-9808bd8d4de5.png height=200 hspace='20'/>
</div>

示例中定义了`作品名`、`人物名`和`时间`三种 Span 类型标签，以及`歌手`、`发行时间`和`所属专辑`三种 Relation 标签。Relation 标签**由 Subject 对应实体指向 Object 对应实体**。

该标注示例对应的 schema 为：

```text
schema = {
    '作品名': [
        '歌手',
        '发行时间',
        '所属专辑'
    ]
}
```

#### 5.3 事件抽取

事件抽取 (Event Extraction, 简称 EE)，是指从自然语言文本中抽取事件并识别事件类型和事件论元的技术。UIE 所包含的事件抽取任务，是指根据已知事件类型，抽取该事件所包含的事件论元。

标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167248793-138a1e37-43c9-4933-bf89-f3ac7228bf9c.png height=200 hspace='20'/>
</div>

示例中定义了`地震触发词`（触发词）、`等级`（事件论元）和`时间`（事件论元）三种 Span 标签，以及`时间`和`震级`两种 Relation 标签。触发词标签**统一格式为`XX 触发词`**，`XX`表示具体事件类型，上例中的事件类型是`地震`，则对应触发词为`地震触发词`。Relation 标签**由触发词指向对应的事件论元**。

该标注示例对应的 schema 为：

```text
schema = {
    '地震触发词': [
        '时间',
        '震级'
    ]
}
```


<a name="数据导出"></a>

## 6. 数据导出

#### 6.1 导出抽取式任务数据

选择导出的文件类型为``JSONL(relation)``，导出数据示例：

```text
{
    "id": 38,
    "text": "百科名片你知道我要什么，是歌手高明骏演唱的一首歌曲，1989年发行，收录于个人专辑《丛林男孩》中",
    "relations": [
        {
            "id": 20,
            "from_id": 51,
            "to_id": 53,
            "type": "歌手"
        },
        {
            "id": 21,
            "from_id": 51,
            "to_id": 55,
            "type": "发行时间"
        },
        {
            "id": 22,
            "from_id": 51,
            "to_id": 54,
            "type": "所属专辑"
        }
    ],
    "entities": [
        {
            "id": 51,
            "start_offset": 4,
            "end_offset": 11,
            "label": "作品名"
        },
        {
            "id": 53,
            "start_offset": 15,
            "end_offset": 18,
            "label": "人物名"
        },
        {
            "id": 54,
            "start_offset": 42,
            "end_offset": 46,
            "label": "作品名"
        },
        {
            "id": 55,
            "start_offset": 26,
            "end_offset": 31,
            "label": "时间"
        }
    ]
}
```

标注数据保存在同一个文本文件中，每条样例占一行且存储为``json``格式，其包含以下字段
- ``id``: 样本在数据集中的唯一标识 ID。
- ``text``: 原始文本数据。
- ``entities``: 数据中包含的 Span 标签，每个 Span 标签包含四个字段：
    - ``id``: Span 在数据集中的唯一标识 ID。
    - ``start_offset``: Span 的起始 token 在文本中的下标。
    - ``end_offset``: Span 的结束 token 在文本中下标的下一个位置。
    - ``label``: Span 类型。
- ``relations``: 数据中包含的 Relation 标签，每个 Relation 标签包含四个字段：
    - ``id``: (Span1, Relation, Span2)三元组在数据集中的唯一标识 ID，不同样本中的相同三元组对应同一个 ID。
    - ``from_id``: Span1对应的标识 ID。
    - ``to_id``: Span2对应的标识 ID。
    - ``type``: Relation 类型。


<a name="数据转换"></a>

## 7.数据转换

该章节详细说明如何通过`doccano.py`脚本对 doccano 平台导出的标注数据进行转换，一键生成训练/验证/测试集。

#### 7.1 抽取式任务数据转换

- 当标注完成后，在 doccano 平台上导出 `JSONL(relation)` 形式的文件，并将其重命名为 `doccano_ext.json` 后，放入 `./data` 目录下。
- 通过 [doccano.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/application/information_extraction/doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --save_dir ./data \
    --negative_ratio 1
```

可配置参数说明：

- ``doccano_file``: 从 doccano 导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，目前只有信息抽取这一种任务。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为 True。
- ``seed``: 随机种子，默认为1000.
- ``schema_lang``: 选择 schema 的语言，可选有`ch`和`en`。默认为`ch`，英文数据集请选择`en`。

备注：
- 默认情况下 doccano.py 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 doccano.py 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从 doccano 导出的文件，默认文件中的每条数据都是经过人工正确标注的。

## References
- **[doccano](https://github.com/doccano/doccano)**
