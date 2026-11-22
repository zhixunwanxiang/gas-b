"""Create a landscape PDF, following the supplied closed-loop report structure."""
# 历史闭环报告生成器：汇总已有回放证据与图表，不发起平台训练或真实模型调用。
# 报告中的分数和部署来自所选运行目录，不能用来覆盖新版本真实测试结果。
import argparse
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, dates as md, font_manager
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from PIL import Image as PILImage

CN = ["乙醇", "乙烯", "氨气", "乙醛", "丙酮", "甲苯"]
CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
TEAL, ORANGE, RED, GRAY, NAVY = "#087f8c", "#ad6b24", "#c76255", "#8998a8", "#245064"
REASONS = {"all_gates_passed": "全部门槛通过", "below_reference_validation": "低于固定验证基线",
           "fixed_f1_regression": "固定验证F1退化", "fixed_accuracy_regression": "固定验证准确率退化",
           "zero_fixed_acetaldehyde_recall": "固定验证乙醛零召回", "fixed_acetaldehyde_regression": "固定验证乙醛召回退化",
           "current_gain_below_minimum": "本窗F1增益不足0.002", "recent_gain_below_margin": "近三窗加权增益不足0.005",
           "current_accuracy_regression": "本窗准确率下降超过0.02", "current_acetaldehyde_regression": "本窗乙醛召回下降超过0.02"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stamp(value):
    return datetime.fromisoformat(value).strftime("%H:%M:%S.%f")[:-3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/closed-loop-four-sensor-validation.pdf")
    args = parser.parse_args()
    run, target = args.run.resolve(), args.output.resolve()
    summary, windows = read(run / "summary.json"), read(run / "history.json")
    spans, candidates, verification, policy = [read(run / name) for name in ("phase-timing.json", "candidates.json", "verification.json", "policy.json")]
    assert verification["passed"]
    assert all(np.array_equal(np.load(run / "windows" / f"batch{b:02d}/predictions.npy", allow_pickle=False),
                              np.load(run / "windows" / f"batch{b:02d}/packaged-control.npy", allow_pickle=False)) for b in range(3,11))
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc").get_name(),
                         "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=190, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    x = np.arange(3,11)
    fig, ax = plt.subplots(2, 1, figsize=(12.8,6.2), gridspec_kw={"height_ratios":[2.1,1]})
    for key, label, color, fmt in [("previous_pdf","参考PDF上轮",ORANGE,"--s"), ("operational","本次闭环",TEAL,"-o")]:
        values = [w[key]["macro_f1"]*100 for w in windows]
        ax[0].plot(x, values, fmt, color=color, label=label, linewidth=1.8, markersize=5)
    for b,w in zip(x,windows):
        ax[0].annotate(f'{w["operational"]["macro_f1"]*100:.2f}',(b,w["operational"]["macro_f1"]*100),xytext=(0,7),textcoords="offset points",ha="center",fontsize=9)
    ax[0].set(xticks=x,xticklabels=[f"dat{b}" for b in x],ylim=(0,100),ylabel="固定六类 Macro-F1（%）")
    ax[0].legend(loc="upper right",ncols=2,frameon=False)
    ax[0].text(.02,.06,"当前项目原回放与本次闭环逐条预测一致，曲线重合。",transform=ax[0].transAxes,color=GRAY,fontsize=9)
    delta=[(w["operational"]["macro_f1"]-w["previous_pdf"]["macro_f1"])*100 for w in windows]
    bars=ax[1].bar(x,delta,color=[TEAL if d>=0 else RED for d in delta],width=.55)
    ax[1].bar_label(bars,labels=[f"{d:+.2f}" for d in delta],padding=3,fontsize=9)
    ax[1].axhline(0,color=NAVY,linewidth=.7)
    ax[1].set(xticks=x,xticklabels=[f'dat{w["batch"]}\nn={w["rows"]:,}' for w in windows],ylim=(-4,27),ylabel="较参考上轮变化\n（百分点）")
    for a in ax:a.grid(axis="y",alpha=.15);a.set_axisbelow(True)
    fig.tight_layout(h_pad=1.8);save(fig,"01-forward-performance")

    fig,ax=plt.subplots(figsize=(12.8,6.0))
    phase_defs=[("drift_monitor",-.25,"#487fb3","o","漂移监测"),("diagnosis",-.15,"#8369a3","D","诊断归因"),
                ("rollback_check",-.05,RED,"s","历史回滚检查"),("experiment_training",.05,ORANGE,"^","候选训练"),
                ("evaluation_gate",.15,TEAL,"P","评估门槛"),("deployment_switch",.25,NAVY,"X","实际切换")]
    for phase,off,color,marker,label in phase_defs:
        used=False
        for s in spans:
            if s["phase"] != phase:continue
            lo,hi=[md.date2num(datetime.fromisoformat(s[k])) for k in ("started_at","ended_at")]
            yy=s["batch"]-3+off
            if s["wall_seconds"]>=.08:
                ax.plot([lo,hi],[yy,yy],color=color,linewidth=4,label=None if used else label)
            else:ax.scatter([lo],[yy],color=color,marker=marker,s=22,label=None if used else label,zorder=4)
            used=True
    for yy,phase,label in [(8,"isolated_integrity_rollback","损坏回滚演练"),(9,"isolated_performance_rollback","性能回滚演练")]:
        s=next(s for s in spans if s["phase"]==phase)
        ax.scatter([datetime.fromisoformat(s["ended_at"])],[yy],color=RED,marker="*",s=75)
    ax.set(yticks=range(10),yticklabels=[f"dat{b}"for b in x]+["损坏演练","性能演练"],ylim=(9.6,-.6),xlabel="北京时间（2026-11-21，Asia/Shanghai）")
    ax.xaxis.set_major_formatter(md.DateFormatter("%H:%M:%S",tz=datetime.fromisoformat(spans[0]["started_at"]).tzinfo))
    ax.xaxis.set_major_locator(md.SecondLocator(interval=1))
    ax.grid(axis="x",alpha=.18)
    ax.legend(ncols=3,loc="upper center",bbox_to_anchor=(.5,1.16),frameon=False,fontsize=9)
    fig.tight_layout();save(fig,"02-lifecycle-timeline")

    fig,ax=plt.subplots(2,1,figsize=(12.8,6.1),gridspec_kw={"height_ratios":[.85,1.6]})
    levels=[0 if w["prediction_version"]=="start_only" else 1 for w in windows]
    ax[0].step(x,levels,where="post",color=TEAL,linewidth=2,label="本批实际模型")
    ax[0].scatter(x+.12,[0 if w["next_version"]=="start_only" else 1 for w in windows],marker="D",facecolors="none",edgecolors=ORANGE,label="本批后选择")
    ax[0].set(xticks=x,xticklabels=[f"dat{b}" for b in x],yticks=[0,1],yticklabels=["起点 start","重训四传感器路由"],ylim=(-.35,1.4),title="dat3后激活；dat7起路由统计开始影响预测，后续6个重训方案均未晋升")
    ax[0].legend(loc="lower right",ncols=2,frameon=False,fontsize=9)
    xx=np.arange(6)
    for off,key,color,label in [(-.19,"previous_pdf",ORANGE,"参考上轮"),(.19,"current",TEAL,"本次闭环")]:
        vals=np.asarray(summary[key]["recall"])*100
        bars=ax[1].bar(xx+off,vals,width=.35,color=color,label=label)
        ax[1].bar_label(bars,labels=[f"{v:.1f}" for v in vals],padding=3,fontsize=8.5)
    ax[1].set(xticks=xx,xticklabels=CN,ylim=(0,108),ylabel="召回率（%）",title="乙醛与甲苯退化：总体F1提升并不代表每个类别改善")
    ax[1].legend(ncols=2,loc="upper right",frameon=False,fontsize=9)
    ax[1].grid(axis="y",alpha=.15);ax[1].set_axisbelow(True)
    fig.tight_layout(h_pad=2.1);save(fig,"03-deployment-recall")

    fig,ax=plt.subplots(figsize=(12.8,4.6))
    values=np.array([w["drift"]["sensor_psi"] for w in windows])
    im=ax.imshow(np.minimum(values,5),aspect="auto",cmap="YlOrBr",vmin=0,vmax=5)
    ax.set(xticks=range(16),xticklabels=[str(i) for i in range(1,17)],yticks=range(8),yticklabels=[f'dat{w["batch"]} ({w["drift"]["flagged_features"]}/128)'for w in windows],xlabel="传感器编号（每个8维特征的PSI均值）")
    for r in range(8):
        for c in range(16):
            ax.text(c,r,f"{values[r,c]:.1f}",ha="center",va="center",fontsize=7.5,color="white" if values[r,c]>3 else NAVY)
    fig.colorbar(im,ax=ax,fraction=.025,pad=.02,label="平均PSI；颜色上限5，格内为原值")
    fig.tight_layout();save(fig,"04-drift-heatmap")

    pdfmetrics.registerFont(TTFont("YaHei",r"C:\Windows\Fonts\msyh.ttc",subfontIndex=0))
    pdfmetrics.registerFont(TTFont("YaHei-Bold",r"C:\Windows\Fonts\msyhbd.ttc",subfontIndex=0))
    pdfmetrics.registerFontFamily("YaHei",normal="YaHei",bold="YaHei-Bold",italic="YaHei",boldItalic="YaHei-Bold")
    W,H=landscape(A4);C=W-72;story=[]
    styles={"title":ParagraphStyle("Title",fontName="YaHei-Bold",fontSize=20,leading=27,textColor=colors.HexColor("#153b4d"),spaceAfter=9,wordWrap="CJK"),
            "body":ParagraphStyle("Body",fontName="YaHei",fontSize=9.6,leading=14,spaceAfter=7,wordWrap="CJK"),
            "small":ParagraphStyle("Small",fontName="YaHei",fontSize=8,leading=11.5,spaceAfter=5,wordWrap="CJK"),
            "cell":ParagraphStyle("Cell",fontName="YaHei",fontSize=7.7,leading=10.5,wordWrap="CJK"),
            "head":ParagraphStyle("Head",fontName="YaHei-Bold",fontSize=8.1,leading=11,textColor=colors.white,wordWrap="CJK")}

    def paragraph(s,kind="body"):return Paragraph(str(s),styles[kind])
    def add(s,kind="body"):story.append(paragraph(s,kind))
    def title(s,sub=""):
        add(s,"title")
        if sub:add(sub,"small")
    def page():story.append(PageBreak())
    def table(head,rows,widths):
        t=Table([[paragraph(escape(str(v)),"head")for v in head]]+[[paragraph(str(v),"cell")for v in row]for row in rows],colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("VALIGN",(0,0),(-1,-1),"TOP"),
                              ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
                              ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                              ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f6f8")]),
                              ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#dbe4e8"))]))
        story.extend([t,Spacer(1,6)])
    def figure(name,maxh):
        path=figures/f"{name}.png"
        with PILImage.open(path)as im:w,h=im.size
        sc=min(C/w,maxh/h);story.append(Image(str(path),width=w*sc,height=h*sc,hAlign="CENTER"))
    def span_text(batch,phase):
        s=next((s for s in spans if s["batch"]==batch and s["phase"]==phase),None)
        return f'{stamp(s["started_at"])}<br/>{stamp(s["ended_at"])}' if s else "未执行"
    def reasons(candidate):return "；".join(REASONS.get(k,k)for k in candidate["reasons"])
    def params(spec):
        entries=list(spec.items());rows=[]
        def val(v):return escape(json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else str(v))
        for i in range(0,len(entries),2):
            a=entries[i];b=entries[i+1]if i+1<len(entries)else("","")
            rows.append([escape(a[0]),val(a[1]),escape(b[0]),val(b[1])])
        table(["参数","实际配置","参数","实际配置"],rows,[120,C/2-120,120,C/2-120])

    current,old,control=[summary[k]for k in ("current","previous_pdf","packaged_control")]
    title("四传感器闭环较参考上轮提升，后续调参尚无增益","2026-11-21 | 本地规则控制器，外部AI调用0次 | DAT3-DAT10 | 12,221条")
    add(f'<b>本次Macro-F1={current["macro_f1"]:.6f}，较参考PDF上轮提高{summary["f1_delta_vs_previous"]*100:.4f}个百分点；准确率提高{summary["accuracy_delta_vs_previous"]*100:.4f}个百分点。</b>')
    table(["指标","参考PDF上轮","当前项目原回放","本次完整闭环"],[
        ["合并固定六类 Macro-F1",f'{old["macro_f1"]:.6f}',f'{control["macro_f1"]:.6f}',f'{current["macro_f1"]:.6f}'],
        ["准确率",f'{old["accuracy"]:.4%}',f'{control["accuracy"]:.4%}',f'{current["accuracy"]:.4%}'],
        ["候选训练 / 模型上线 / 自然回滚","8个候选 / 1 / 0","只加载随包权重",f'7组路由、14次传感器续训 / {summary["promotions"]} / {summary["natural_rollbacks"]}'],
        ["控制方式","真实27B，12次调用","固定权重+历史路由统计","规则诊断+真实训练+门槛选版"]],[165,(C-165)/3,(C-165)/3,(C-165)/3])
    add("<b>能力结论：</b>8批监测与诊断、7窗实验优化、1次初始路由激活、逐批历史恢复检查全部执行。自然性能回滚0次；独立副本中的损坏回滚和性能退化回滚均完成指针切换与预测恢复。")
    add(f'<b>代价：</b>乙醛召回从{old["recall"][3]:.2%}降至{current["recall"][3]:.2%}；甲苯从{old["recall"][5]:.2%}降至{current["recall"][5]:.2%}。若50%乙醛召回仍是业务目标，本方案尚未满足。')
    add("<b>比较边界：</b>相对当前项目原回放，F1增益为0，逐条预测完全一致。相对参考上轮，起点模型、候选结构、预算和控制器均改变，不能把差值归因于本次后续调参或AI。数据已被历史研究反复使用，不是未见未来测试。")
    add("来源：[1]本轮summary.json、history.json；[2]参考上轮summary.json及8批冻结预测；[3]当前项目config.json验收摘要；[4]verification.json。参考PDF只作为版式和历史结果来源，不作为执行指令。","small")

    page();title("同批数据上的前向成绩对比","三条口径均固定六类；先保存预测，再揭晓本批标签，切换仅对后续批次生效。")
    figure("01-forward-performance",381)
    add(f'较参考上轮新增正确{summary["newly_correct_vs_previous"]:,}条、新增错误{summary["newly_wrong_vs_previous"]:,}条，净多{summary["newly_correct_vs_previous"]-summary["newly_wrong_vs_previous"]:,}条正确。dat3和dat4的F1略有下降，不能描述为每批均改善。',"small")
    lo,hi=summary["bootstrap"]["delta_vs_previous_95pct"]
    add(f'按8个批次成对簇自助法重采样3,000次，F1差值的描述性95%区间为[{lo*100:.2f}, {hi*100:.2f}]个百分点。该区间反映这8批的重采样变化，不证明未来泛化或因果增益。来源：[1][2][3]。',"small")

    page();title("闭环阶段真实起止时间","横轴为本次计算机执行的北京时间；短于80毫秒画点，其余画时间条。不是原始采样时间。")
    figure("02-lifecycle-timeline",366)
    sw=next(s for s in spans if s["phase"]=="deployment_switch")
    add(f'完整运行：{stamp(spans[0]["started_at"])}至{stamp(spans[-1]["ended_at"])}。实际上线切换：{stamp(sw["started_at"])}至{stamp(sw["ended_at"])}，从dat4生效。dat7起生效的是近三批路由统计更新，不是新增模型晋升。',"small")
    add("监测 → 保存预测 → 揭晓标签评分 → 诊断 → 历史回滚检查 → 训练候选 → 评估门槛 → 通过时切换；末批跳过训练。两项回滚演练在正式流结束后的隔离副本执行。来源：[1]phase-timing.json、events.jsonl。","small")

    page();title("逐批阶段时间明细","2026-11-21，Asia/Shanghai。每格上行为开始，下行为结束，显示到毫秒；同值不代表未执行。")
    phases=["drift_monitor","diagnosis","experiment_training","evaluation_gate","rollback_check","deployment_switch"]
    table(["批次","漂移监测","诊断归因","候选训练","评估门槛","历史回滚检查","实际上线"],[[f'dat{b}',*[span_text(b,p)for p in phases]]for b in x],[42]+[(C-42)/6]*6)
    for phase,name in [("isolated_integrity_rollback","隔离损坏回滚"),("isolated_performance_rollback","隔离性能回滚")]:
        s=next(s for s in spans if s["phase"]==phase)
        add(f'{name}：{stamp(s["started_at"])}至{stamp(s["ended_at"])}；正式部署指针保持不变。',"small")
    training=sum(s["wall_seconds"]for s in spans if s["phase"]=="experiment_training")
    gate=sum(s["wall_seconds"]for s in spans if s["phase"]=="evaluation_gate")
    add(f'7窗候选训练阶段合计{training:.3f}秒；评估门槛阶段合计{gate:.3f}秒。后者包含仅重放已结束历史的影子评估，不能计作未来前向收益。完整微秒记录与CPU耗时见phase-timing.csv。',"small")

    page();title("候选来源、成绩与上线决定","全部候选来自预先锁定的程序配方，实际训练；没有27B或其他外部AI建议。分数为事后影子验证。")
    table(["批次 / 方案","固定验证F1","本窗影子F1","近期差值","上线","决定依据"],[[
        f'dat{c["batch"]} / '+("原配方重训"if c["cold_start"]else CN[CLASSES.index(windows[c["batch"]-3]["diagnosis"]["weakest_class"])]+"加权续训"),
        f'{c["fixed"]["macro_f1"]:.6f}',f'{c["current_shadow"]["macro_f1"]:.6f}',f'{c["recent_f1"]-c["incumbent_recent_f1"]:+.6f}',
        "是"if c["selected"]else"否",reasons(c)]for c in candidates],[155,85,85,85,38,C-448])
    add("<b>初始激活与后续晋升的区别：</b>dat3之后采用冷启动门槛：固定验证F1至少提高0.002、准确率最多下降0.02、乙醛召回非零；dat3影子分仍是起点模型，不声称本批已获益。此后新增方案还须本窗F1提升0.002及近期加权提升0.005。")
    add("<b>为何拒绝：</b>dat4虽本窗F1提升约0.00264，但近期优势不足；dat9本窗影子分也提高，仍未达到近期门槛。dat8另外触发乙醛召回退化保护。未通过候选保留检查点与证据，但不改变下一批实际版本。")
    add("近期分是最近三批F1按0.6/0.3/0.1加权；不足三批时重新归一。它仅用于选择，最终成绩另由全部冻结预测的合并混淆矩阵计算。来源：[1]candidates.json、policy.json。","small")

    page();title("实际版本与类别代价","模型晋升与路由统计更新分别记录；总体分数提高仍伴随乙醛、甲苯召回下降。")
    figure("03-deployment-recall",389)
    add(f'氨气召回从{old["recall"][2]:.2%}提高到{current["recall"][2]:.2%}，丙酮从{old["recall"][4]:.2%}提高到{current["recall"][4]:.2%}；乙醛和甲苯退化见图。选择门槛防的是新候选相对当前版本退化，不保证活动模型跨批漂移时不退化。来源：[1][2]。',"small")

    page();title("漂移持续存在，诊断属于待验证归因","PSI、KS经验CDF距离、均值偏移基于128维特征，与固定batch1训练集比较；只读本批特征。")
    figure("04-drift-heatmap",289)
    add("<b>告警规则：</b>至少30条样本，任一维PSI&gt;0.2、KS&gt;0.2或均值偏移&gt;1σ即告警；8批均触发。直方图以训练分位数划分，概率下限为10^-6。格内为每个传感器8维PSI均值，颜色上限5。")
    add("<b>归因步骤：</b>本批预测冻结、标签揭晓后，统计支持数≥20的类别召回、最大错分方向、传感器PSI前三名及类别比例。选最低召回类别加强公开训练中的权重。第13号传感器另有因果平线检测，报警样本由四传感器分支兜底；该结构来自已有配方。")
    add("<b>可支持的结论：</b>测到了分布变化和具体错分；没有硬件日志、浓度或温湿度对照，不能确定传感器老化、物理故障或环境变化是原因。类比例变化也可能贡献PSI。来源：[1]windows/batchXX/drift.json、diagnosis.json；原router.py、signals.py。","small")

    page();title("五段闭环怎样选择，以及实际使用的配方")
    for h,text in [
        ("1 漂移监测","逐批只读特征计算PSI/KS/均值偏移；达到阈值开启优化，未读取未来批次标签。"),
        ("2 诊断归因","冻结后揭晓标签，找有足够支持的最低召回类别及主要混淆；这是优化假设，不是物理因果结论。"),
        ("3 实验优化","dat3重训原配方的home分支及四/五传感器；dat4-9从当前实际四/五传感器父检查点续训，每窗两次，保持16个传感器、128维的输入契约。梯度仅batch1。"),
        ("4 评估上线","batch2选epoch、固定验证保底。后续候选还需本窗F1增益≥0.002、近三窗加权增益≥0.005、准确率下降≤0.02；固定验证及有足够支持的本窗乙醛召回下降≤0.02。通过后原子替换本地部署指针，下一批生效。"),
        ("5 回滚恢复","比较曾上线历史版本，近期优势≥0.005且通过固定验证保底、F1/准确率/乙醛保护才恢复。正式流无自然回滚；隔离演练另外验证了损坏检测、同一性能门槛和实际恢复。")]:add(f'<b>{h}</b> {text}',"small")
    add("<b>初始传感器配方：</b>learning_rate=0.003，epochs=24，minimum_epochs=16，balance=0.5，weak_boost=1.5，乙醛class_boost=2，l2=0.001，batch_size=128。四传感器保留[3,14,15,16]，五传感器保留[3,13,14,15,16]；两者均从reference.npz完整状态续训。","small")
    add("<b>home配方：</b>从start.npz按ID划分80%训练/20%校准；输出层单轮Adam，learning_rate=0.00065、balance=0.35、l2=0.0002、label_smoothing=0.02；训练内Fisher取120维KNN，k=3、distance_power=3、class_balance=0.25、MLP混合0.4；每个二分类头取8维。阈值0.5/0.65由原训练集五折校准确定，固定使用。","small")
    add("<b>后续续训：</b>learning_rate=0.0005，8轮、最少4轮，balance=0.35、weak_boost=1.5、l2=0.001、anchor_l2=0.01、batch_size=128；当窗最弱类class_boost=1.5。最佳epoch与实际执行量分别保存。数据：batch1 445条，batch2 1,244条；回放12,221条均不参与梯度。","small")
    add("本地部署不包含HTTP服务、平台SDK适配、云上传或真实27B决策；不能将本轮规则流程等同于参考PDF的AI自主闭环。来源：[1]policy.json、registry.json、各training/experiment.json。","small")

    for w in windows:
        b=w["batch"];c=w["candidate"];d=w["diagnosis"]
        page();title(f'dat{b} | 诊断、实验参数与选择证据',f'本批实际版本：{w["prediction_version"]}；结束后选择：{w["next_version"]}；仅影响后续批次。')
        add(f'<b>本批结果：</b>n={w["rows"]:,}；固定六类F1={w["operational"]["macro_f1"]:.6f}；参考上轮={w["previous_pdf"]["macro_f1"]:.6f}；准确率={w["operational"]["accuracy"]:.2%}。',"small")
        conf=d["top_confusion"]
        add(f'<b>诊断与假设：</b>{w["drift"]["flagged_features"]}/128维告警，传感器PSI前三名{d["top_drift_sensors"]}；有足够支持的最弱类是{CN[CLASSES.index(d["weakest_class"])]}，召回{d["weakest_recall"]:.2%}。最大错分为{CN[CLASSES.index(conf["true"])]}→{CN[CLASSES.index(conf["predicted"])]}，{conf["rows"]}条。假设存在分布漂移与类间混淆，以公开训练的类别加权进行验证。',"small")
        if c:
            add("<b>实际参数：</b>两次传感器训练除名称/保留传感器外使用共同配置；home分支仅dat3重训，见第8页。","small")
            common={k:v for k,v in c["components"][0]["parameters"].items() if k not in ("name","kept_sensors")}
            params(common)
            table(["组件 / 保留传感器","实际父摘要（前12位）","选定累计epoch / 本次执行轮数","组件固定验证F1"],[[
                comp["role"]+" / "+str(comp["parameters"]["kept_sensors"]),comp["parent_sha256"].replace("sha256:","")[:12],
                f'{comp["selected_epoch"]} / {comp["executed_epochs"]}',f'{comp["validation_f1"]:.6f}']for comp in c["components"]],[230,140,220,C-590])
            table(["路由整体评估","当前版本","候选","门槛"],[
                ["固定验证F1",f'{c["incumbent_fixed"]["macro_f1"]:.6f}',f'{c["fixed"]["macro_f1"]:.6f}',"冷启动增益≥0.002；后续下降≤0.02"],
                ["本窗影子F1",f'{c["incumbent_current_shadow"]["macro_f1"]:.6f}',f'{c["current_shadow"]["macro_f1"]:.6f}',"后续新增候选增益≥0.002"],
                ["近期加权F1",f'{c["incumbent_recent_f1"]:.6f}',f'{c["recent_f1"]:.6f}',"后续新增候选增益≥0.005"]],[145,120,120,C-385])
            add(f'<b>最终决定：</b>{"上线，从下一批生效"if c["selected"]else"拒绝新增候选，保留当前版本"}；{reasons(c)}。本窗影子分由已结束数据重放计算，未用于回改冻结预测。',"small")
        else:
            add("<b>实验与上线：</b>本批是验证范围末批，没有后续批次，因此按预定规则跳过训练和新模型晋升；不是训练失败。仍完成监测、冻结评分、诊断、历史回滚检查和路由统计更新。")
            add("<b>终局业务判断：</b>总体Macro-F1较参考上轮改善，当前项目基线保持；乙醛召回不足，不能据此宣布业务指标全面达标。")
            table(["类别","本次总体召回","参考上轮召回","变化（百分点）"],[[CN[k],f'{current["recall"][k]:.2%}',f'{old["recall"][k]:.2%}',f'{(current["recall"][k]-old["recall"][k])*100:+.2f}']for k in range(6)],[140,170,170,C-480])
        state=w["route_state"]
        pairs=[CN[k//6]+"/"+CN[k%6]for k in state["active_pairs"]]
        add(f'<b>路由/回滚：</b>窗口结束后的外层激活类别对：{"、".join(pairs)or"无"}；{state["nondefault_weights"]}个内部权重偏离默认0.75；历史回滚选择{w["rollback"]["selected"]}，本批自然回滚{"发生"if w["rollback"]["switched"]else"未发生"}。',"small")
        table(["监测起止","诊断起止","训练起止","评估起止"],[[span_text(b,p)for p in ("drift_monitor","diagnosis","experiment_training","evaluation_gate")]],[C/4]*4)
        add(f'来源：[1]windows/batch{b:02d}/下的prediction_commit、drift、diagnosis、candidate（如有）、next-state及result文件；训练父摘要和完整参数另保存在该版本training/。',"small")

    page();title("验证、来源与结论适用范围","证据可回放；不以图表代替时序、数据边界和产物完整性检查。")
    add(f'<b>{verification["check_count"]}项检查全部通过。</b>逐批从保存的JSON状态恢复后重新预测，8批逐条一致；14次传感器候选训练及2个home组件从真实父检查点重训，文件SHA-256精确匹配本轮产物。手工逐行累计混淆矩阵，独立复核总体F1、准确率及参考上轮成绩。')
    add("<b>时序与状态：</b>8批均满足监测先于预测保存、预测保存先于揭晓、揭晓先于诊断/训练；下一批实际版本等于上一批选定版本。原路由的重复预测、非法反馈拒绝及有效反馈推进检查通过。原项目算法、数据、随包模型、参考PDF和上轮预测的保护摘要未改变。")
    dr=summary["rollback_drills"]
    add(f'<b>两项隔离回滚：</b>损坏演练向副本状态追加字节，摘要异常被检出并恢复健康指针；性能演练人为将副本切到start_only，使用真实已结束近三批分数{dr["performance"]["observed_degraded_recent_f1"]:.6f}与{dr["performance"]["observed_recovery_recent_f1"]:.6f}及固定验证门槛，恢复四传感器路由。两项均验证恢复后预测一致，正式部署指针未改动；不计作自然漂移回滚。')
    add("<b>重建与浮点：</b>本轮Windows训练的文件摘要与随包四个训练组件存在浮点末位差异，因此新实验用实际生成文件的独立清单绑定；未放宽或修改原config.json。重训方案的8批预测全部通过随包预测摘要验收，并与原回放一致。")
    add("<b>限制：</b>1）历史数据反复研究，参数结构含历史研究结果；本次只保证运行时没有未来标签泄漏，不构成全新留出集。2）与参考上轮初始检查点、预算及控制器不同；总体差值是配置组合比较。3）没有采集时间戳，图表只能展示dat批序和本次计算机执行时间。4）上线是本地部署指针和状态切换，未验证线上HTTP、SDK或真实AI服务。5）后续6个新方案未晋升，不能说持续调参已进一步提升原项目。","small")
    table(["来源编号","可复核证据"],[
        ["[1] 本轮",escape(str(run))+"；summary.json / history.json / policy.json / registry.json / events.jsonl / phase-timing.csv"],
        ["[2] 参考上轮",escape(summary["previous_run"])+"；原PDF、summary.json及windows/batchXX/predictions.npz"],
        ["[3] 当前项目",escape(str(ROOT))+"；config.json、models/、run.py及src/gas_a/"],
        ["[4] 独立核验","verification.json、verification/training-replay/、rollback_drills/summary.json、protected-hashes-before/after.json"]],[110,C-110])
    add("复现：先运行scripts/closed_loop_validation.py（--dataset、--previous、--output），再运行scripts/verify_closed_loop.py，最后用独立.venv-report环境运行scripts/build_closed_loop_report.py。推理依赖与报告依赖分别见requirements.txt及requirements-report.txt。","small")

    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#dbe4e8"));canvas.line(36,31,W-36,31)
        canvas.setFont("YaHei",7.3);canvas.setFillColor(colors.HexColor("#546b78"))
        canvas.drawString(36,19,"Gas A | DAT3-DAT10 | 2026-11-21 | 四传感器本地闭环验证")
        canvas.drawRightString(W-36,19,str(doc.page));canvas.restoreState()
    target.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(target),pagesize=(W,H),rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=43,
                          title="四传感器模型漂移闭环验证与上轮对比",author="Local validation")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    reader=PdfReader(target)
    text="\n".join(p.extract_text()for p in reader.pages)
    required=["0.649151","0.585774","6.3377","100项检查","自然性能回滚0次","乙醛"]
    for value in required:assert value in text,value
    (run/"report-generation.json").write_text(json.dumps({"pdf":str(target),"pages":len(reader.pages),"required_phrases_found":required,"source_run":str(run)},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"pdf":str(target),"pages":len(reader.pages)},ensure_ascii=False))


if __name__=="__main__":main()
