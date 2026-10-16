"""GPT proposes disposable trials; offline validation owns acceptance."""
import json
import os
import urllib.request
import urllib.error
from .candidates import deletable_turns
from .vision import build_user_content

PROMPT = '''你正在审查一段三维场景构建轨迹，目的是找出其中冗余、不必要的试错步骤，将它压缩成一条高效、连贯的纠错轨迹。

模型在构建场景时，可能需要反复移动、旋转、缩放或吸附物体，才能解决碰撞、位置不合理、遮挡、通行受阻等问题。部分尝试是完成纠错所必需的，另一些只是无效尝试、重复操作或随后被完全替代的调整。

你的任务是逐步判断“保留”或“删除”。目标不是尽可能多地删除，而是在保持纠错结果和轨迹连续性的前提下，去掉不必要的中间试错。

一、你会收到哪些输入

你会收到一个轨迹片段的结构化信息，以及与步骤对应的场景图片。

结构化信息包括：

1. 片段信息：所属构建阶段、片段编号、起止步骤。
2. operations：按原始顺序排列的场景操作。每个操作包含步骤编号、工具名称、作用对象、调用参数、工具返回结果和保护标记。
3. entry_check：片段开始前可用的检查反馈，可能指出需要修复的问题；也可能为空。
4. checks：片段中的检查记录，包括原始反馈，以及通过、失败或未知的解析结果。
5. events：将操作和检查合并后，按实际发生顺序排列的事件。
6. metrics：辅助指标，例如相邻操作的坐标参数变化、是否完全重复。这些指标只能辅助判断，不能单独证明一个步骤可以删除。

图片按步骤顺序提供，每张图片前都有说明：

- turn_index：图片所属步骤，对应原始轨迹中的编号。
- view_index：该步骤内的视角序号，不是步骤编号。
- path：原始图片路径，可辅助识别视角；实际图片也会随输入提供。
- dimensions：原始图片尺寸。
- sha256：用于识别图片文件的摘要。
- relation：说明图片与动作的时间关系。

relation 的含义：

- assistant_input_before_action：模型执行该步骤动作之前看到的场景。不能将它当作该动作执行后的结果。
- following_assistant_input：片段结束后，下一轮模型看到的场景，可作为判断片段出口状态的参考。

一个步骤可能只有俯视图，也可能包含多个视角。属于同一步骤的多张图片共同描述该步骤的输入场景，请结合观察。

工具反馈、检查结果和图片可能存在信息缺失或不一致。遇到这种情况，应在判断理由中指出，不能凭空补全状态，也不能仅凭一张图就断定错误已经修复。

二、片段的首尾边界

这段轨迹的头部对应模型发现错误、开始纠错的位置；尾部对应模型完成纠错的位置。它们是压缩时必须保留的边界。

- operations 中的第一个操作必须保留。
- operations 中的最后一个操作必须保留。
- 标记为 protected 的操作必须保留。
- 边界上的检查记录必须保留。
- 只能对首尾之间的中间操作提出删除建议。
- 如果只有一个或两个操作，则全部保留。

首尾图片、动作和反馈都属于判断上下文，不能因为首尾不允许删除而忽略它们。

如果提供的反馈无法支持“尾部已经完成纠错”，应明确指出这项证据缺口。不要自行修改片段边界，也不要把片段最后一张图片自动解释成纠错成功。

三、最重要的要求：删除后必须保持连续性

将轨迹理解为：

当前状态 → 执行动作 → 下一状态。

删除某些步骤之后，保留下来的轨迹仍必须满足这一关系：下一步的输入状态，必须确实能够由上一保留步骤的输入状态执行其动作得到。

因此，不能仅仅因为最后的场景看起来相似，就判断中间步骤可以删除。你必须同时检查：

1. 保留下来的动作，在删除前序步骤后是否仍然能够执行。
2. 动作使用的对象、资源、坐标系、尺寸、支撑关系等是否仍然有效。
3. 动作是否依赖被删除步骤产生的位置、朝向、几何变化或其他状态。
4. 压缩后的片段能否到达与原片段相同的出口状态。
5. 保留的首尾动作和片段外的后续操作是否仍然能够正确接续。

逐步判断时，要沿着“假设删除这些步骤后”的新状态序列分析，不能继续使用被删步骤产生的旧状态。

四、如何判断中间步骤是否冗余

可以考虑删除的情况包括：

- 一个尝试没有留下后续所需的影响，其状态变化被后面的操作完全替代。
- 一个重复操作没有产生必要的新状态，也没有提供后续所需的依赖。
- 多次探索中的某次调整可以跳过，剩余动作仍能正确执行并到达相同出口状态。
- 某次尝试制造了新的问题，之后又被撤销或替换，且没有为保留步骤提供必要条件。

以下情况不能仅凭表面现象判为冗余：

- 移动距离很小。
- 两张图片看起来相似。
- 同一物体被连续操作多次。
- 一次操作之后检查仍然失败。
- 一个动作的位置参数后来被另一个动作覆盖。

尤其要区分绝对操作与相对操作：

- 后面的绝对位置操作可能覆盖前面的移动，但仍需检查前面的动作是否改变了其他必要状态。
- 相对移动、吸附、对齐以及依赖碰撞或支撑关系的操作，可能依赖执行前的位置、朝向和几何关系，不能直接当作绝对覆盖。
- 缩放、创建、删除、资源生成等变化，通常不能由后面的移动操作替代。

不要仅根据工具名称，就假设检查、观察或物理相关工具没有状态影响。如果工具会触发模拟、自动调整或其他状态变化，必须将这些影响纳入连续性判断；工具语义不明确时，不要自行假定。

一次尝试产生了检查反馈，不代表这次尝试本身必须保留。删除某个中间试错操作后，程序会一并删除紧随它、属于该试错状态的中间 check_physics 及其结果，不替换反馈。片段边界和阶段末尾检查必须保留。保留步骤的观察只能使用与新轨迹状态匹配的已有图片，不能沿用被删状态下的旧反馈。

同样，一次操作改善了部分问题，也不自动意味着它必须保留；需要判断这种改善是否是后续动作完成纠错的必要条件。

五、联合检查所有删除决定

不能只孤立地判断每一步。

完成初步判断后，必须将所有拟删除步骤一起移除，再按顺序检查剩余动作。两个步骤分别看起来可以删除，并不代表它们可以同时删除。

请严格检查：

- 是否误删了首尾或 protected 操作。
- 是否存在依赖被删步骤结果的保留动作。
- 是否把动作前的图片误当作动作后的图片。
- 是否将不同视角误当成不同时间的状态。
- 是否将不同对象或不同坐标系下的参数直接比较。
- 是否保留了纠错所需的几何、尺寸、资源和对象关系。
- 是否能维持片段出口状态及后续轨迹的连续性。

如果现有证据不足以支持某一步删除后仍然连贯，则判断为保留，并简要说明缺少什么证据。不要编造工具执行结果。

你的判断是删除方案，不是实际执行验证结果。后续程序在核验材料可用时，会依据工具语义离线计算保留动作的位姿变化，并与已有状态记录核对；缺少材料时会跳过对应检查并标记未验证，不会重新构建场景或运行物理仿真。你不能声称已经完成了实际执行或物理验证。

六、输出格式

只输出合法 JSON，不要输出 Markdown 或额外说明：

{
  "judgments": [
    {
      "turn_index": 具体步骤编号,
      "decision": "keep 或 delete",
      "reason": "简要说明该步骤的必要依赖，或为什么删除后仍能保持连续性及相同出口状态"
    }
  ]
}

要求：

- 对 operations 中的每个操作恰好给出一次判断，按原始顺序输出。
- 使用输入中的原始 turn_index，不要重新编号。
- decision 只能是 keep 或 delete。
- 第一个操作、最后一个操作和所有 protected 操作必须输出 keep。
- 不输出置信度。
- 理由应说明具体依据，不要只写“冗余”“有用”或“保持一致”。'''

class GPTReviewer:
    def __init__(self, model='gpt-5.5', url='https://api.openai.com/v1/chat/completions', key=None,
                 cache_path=None, max_attempts=3, timeout=120, label=''):
        self.model=model;self.url=url;self.key=key or os.environ.get('OPENAI_API_KEY')
        if not self.key:raise ValueError('OPENAI_API_KEY is required')
        self.cache_path=cache_path;self.max_attempts=max_attempts;self.timeout=timeout;self.label=label
        self.reviews=[]

    def _cache(self, fingerprint, review=None):
        if self.cache_path is None:return None
        import sqlite3
        from pathlib import Path
        path=Path(self.cache_path);path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(path,timeout=30) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS reviews (fingerprint TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            if review is None:
                row=db.execute('SELECT payload FROM reviews WHERE fingerprint=?',(fingerprint,)).fetchone()
                return json.loads(row[0]) if row else None
            db.execute('INSERT OR REPLACE INTO reviews VALUES (?,?)',(fingerprint,json.dumps(review,ensure_ascii=False)))
        return None

    @staticmethod
    def _parse(review, segment):
        rows=review['judgments'];expected={o['turn_index'] for o in segment['operations']}
        if len(rows)!=len(expected) or {r['turn_index'] for r in rows}!=expected:
            raise ValueError('incomplete or duplicated judgments')
        for row in rows:
            if type(row.get('turn_index')) is not int or row.get('decision') not in ('keep','delete'):
                raise ValueError('invalid judgment types')
        allowed=deletable_turns(segment)
        return {r['turn_index'] for r in rows if r['turn_index'] in allowed and r['decision']=='delete'}

    def __call__(self, segment):
        import hashlib
        import time
        manifest=[];attempt_errors=[];review=None;cached=False;fingerprint=None
        try:
            content,manifest=build_user_content(segment)
            body={'model':self.model,'temperature':0,'messages':[
                {'role':'system','content':PROMPT},{'role':'user','content':content}]}
            data=json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
            fingerprint=hashlib.sha256(self.url.encode()+b'\0'+data).hexdigest()
            try:
                review=self._cache(fingerprint)
                if review is not None:result=self._parse(review,segment);cached=True
            except Exception:
                review=None  # Cache failures never discard otherwise usable data.
            if review is None:
                for attempt in range(1,self.max_attempts+1):
                    try:
                        print(f'[gpt] {self.label} {segment["segment_id"]} images={len(manifest)} attempt={attempt}',flush=True)
                        req=urllib.request.Request(self.url,data=data,headers={
                            'Content-Type':'application/json','Authorization':f'Bearer {self.key}'})
                        with urllib.request.urlopen(req,timeout=self.timeout) as response:payload=json.load(response)
                        value=payload['choices'][0]['message']['content'].strip()
                        if value.startswith('```'):value='\n'.join(value.splitlines()[1:-1])
                        proposed=json.loads(value);result=self._parse(proposed,segment);review=proposed
                        break
                    except Exception as exc:
                        error=dict(attempt=attempt,error=type(exc).__name__)
                        if isinstance(exc,urllib.error.HTTPError):error['http_status']=exc.code
                        attempt_errors.append(error)
                        # Authentication/request-size errors need configuration changes.
                        if isinstance(exc,urllib.error.HTTPError) and exc.code in (400,401,403,404,413):break
                        if attempt<self.max_attempts:time.sleep(min(2**(attempt-1),4))
                if review is None:raise ValueError('review attempts exhausted')
                try:self._cache(fingerprint,review)
                except Exception:pass
        except Exception as exc:
            self.reviews.append(dict(segment=segment['segment_id'],status='unavailable',error=type(exc).__name__,
                                     attempts=attempt_errors,image_count=len(manifest),model=self.model))
            print(f'[gpt] {self.label} unavailable {segment["segment_id"]}: {type(exc).__name__}',flush=True)
            return set()
        print(f'[gpt] {self.label} reviewed {segment["segment_id"]} delete={sorted(result)} cached={cached}',flush=True)
        self.reviews.append(dict(segment=segment['segment_id'],status='reviewed',review=review,
                                 image_count=len(manifest),images=manifest,model=self.model,
                                 cached=cached,request_sha256=fingerprint,attempt_errors=attempt_errors))
        return result
