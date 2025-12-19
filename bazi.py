from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    _HAS_ZONEINFO = True
except Exception:
    _HAS_ZONEINFO = False
from lunar_python import Solar, Lunar
import itertools

class BaZiAutomation:
    def __init__(self):
        # --- 1. 基础映射 ---
        self.gan_wuxing = {"甲": "木", "乙": "木", "丙": "火", "丁": "火", "戊": "土", "己": "土", "庚": "金", "辛": "金", "壬": "水", "癸": "水"}
        self.zhi_wuxing = {"子": "水", "丑": "土", "寅": "木", "卯": "木", "辰": "土", "巳": "火", "午": "火", "未": "土", "申": "金", "酉": "金", "戌": "土", "亥": "水"}
        
        # 五行生克关系
        self.wuxing_relation = {
            "生": [("木","火"), ("火","土"), ("土","金"), ("金","水"), ("水","木")],
            "克": [("木","土"), ("土","水"), ("水","火"), ("火","金"), ("金","木")]
        }

        # 地支藏干 (标准)
        self.hidden_stems = {
            "子": ["癸"], "丑": ["己", "癸", "辛"], "寅": ["甲", "丙", "戊"], "卯": ["乙"],
            "辰": ["戊", "乙", "癸"], "巳": ["丙", "戊", "庚"], "午": ["丁", "己"], "未": ["己", "丁", "乙"],
            "申": ["庚", "壬", "戊"], "酉": ["辛"], "戌": ["戊", "辛", "丁"], "亥": ["壬", "甲"]
        }

        # --- 2. 天干关系库 ---
        self.gan_rel_map = {
            "chong": [("甲","庚"), ("乙","辛"), ("丙","壬"), ("丁","癸")], # 冲
            "he": [("甲","己"), ("乙","庚"), ("丙","辛"), ("丁","壬"), ("戊","癸")], # 五合
        }
        # 五合化气 (简版，实际需要看月令，这里只提示意象)
        self.gan_he_hua = {("甲","己"): "土", ("乙","庚"): "金", ("丙","辛"): "水", ("丁","壬"): "木", ("戊","癸"): "火"}

        # --- 3. 地支关系库 ---
        # 六合
        self.zhi_liu_he = {
            frozenset(["子","丑"]): "化土", frozenset(["寅","亥"]): "化木", frozenset(["卯","戌"]): "化火",
            frozenset(["辰","酉"]): "化金", frozenset(["巳","申"]): "化水", frozenset(["午","未"]): "化土/火"
        }
        # 六冲
        self.zhi_liu_chong = [
            frozenset(["子","午"]), frozenset(["丑","未"]), frozenset(["寅","申"]),
            frozenset(["卯","酉"]), frozenset(["辰","戌"]), frozenset(["巳","亥"])
        ]
        # 三合 (半合)
        self.zhi_san_he = {
            "水局": ["申", "子", "辰"], "木局": ["亥", "卯", "未"], 
            "火局": ["寅", "午", "戌"], "金局": ["巳", "酉", "丑"]
        }
        # 三会 (方局)
        self.zhi_san_hui = {
            "水会": ["亥", "子", "丑"], "木会": ["寅", "卯", "辰"],
            "火会": ["巳", "午", "未"], "金会": ["申", "酉", "戌"]
        }
        # 相刑
        self.zhi_xing = {
            "无礼刑": [("子", "卯")],
            "恃势刑": [("寅", "巳"), ("巳", "申"), ("申", "寅")], # 寅巳申循环
            "丑未戌刑": [("丑", "未"), ("未", "戌"), ("戌", "丑")],
            "自刑": ["辰", "午", "酉", "亥"]
        }
        # 相害 (穿)
        self.zhi_hai = [
            frozenset(["子","未"]), frozenset(["丑","午"]), frozenset(["寅","巳"]),
            frozenset(["卯","辰"]), frozenset(["申","亥"]), frozenset(["酉","戌"])
        ]
        # 相破
        self.zhi_po = [
            frozenset(["子","酉"]), frozenset(["午","卯"]), frozenset(["巳","申"]),
            frozenset(["寅","亥"]), frozenset(["辰","丑"]), frozenset(["戌","未"])
        ]

    def get_runtime_year_info(self, as_of: datetime | None = None, tz_name: str = "Asia/Shanghai"):
        """
        返回运行时（按指定时区）的当前公历年份 & 流年（年柱干支）。
        tz_name 默认 Asia/Shanghai（中国时间）
        """
        # 1) 取“当前时间”：强制用指定时区
        if as_of is None:
            if _HAS_ZONEINFO:
                now = datetime.now(ZoneInfo(tz_name))
            else:
                # fallback: pip install pytz
                import pytz
                now = datetime.now(pytz.timezone(tz_name))
        else:
            # 如果你传入了 as_of：确保它带时区；没有就按 tz_name 补上
            if as_of.tzinfo is None:
                if _HAS_ZONEINFO:
                    now = as_of.replace(tzinfo=ZoneInfo(tz_name))
                else:
                    import pytz
                    now = as_of.replace(tzinfo=pytz.timezone(tz_name))
            else:
                now = as_of

        # 2) 用 now 生成 Solar/Lunar/EightChar
        solar_now = Solar.fromYmdHms(
            now.year, now.month, now.day,
            now.hour, now.minute, now.second
        )
        lunar_now = solar_now.getLunar()
        ec_now = lunar_now.getEightChar()
        ec_now.setSect(1)

        ly_gan = ec_now.getYearGan()
        ly_zhi = ec_now.getYearZhi()

        return {
            "now": now,
            "tz_name": tz_name,
            "current_gregorian_year": now.year,
            "current_solar_str": f"{now.year}年{now.month}月{now.day}日 {now.hour:02d}:{now.minute:02d}:{now.second:02d} ({tz_name})",
            "current_lunar_str": lunar_now.toFullString() if hasattr(lunar_now, "toFullString") else str(lunar_now),
            "liu_nian_ganzhi": f"{ly_gan}{ly_zhi}",
            "liu_nian_gan": ly_gan,
            "liu_nian_zhi": ly_zhi,
            "liu_nian_gan_wuxing": self.gan_wuxing.get(ly_gan, ""),
            "liu_nian_zhi_wuxing": self.zhi_wuxing.get(ly_zhi, ""),
        }

    def check_wuxing_ke(self, gan1, gan2):
        """判断天干克"""
        w1, w2 = self.gan_wuxing[gan1], self.gan_wuxing[gan2]
        if (w1, w2) in self.wuxing_relation["克"]: return f"{gan1}{gan2}相克"
        if (w2, w1) in self.wuxing_relation["克"]: return f"{gan2}{gan1}相克"
        return None

    def check_an_he(self, zhi1, zhi2):
        """
        检查地支暗合 (地支藏干之间的相合)
        定义：地支本气或藏干与另一地支藏干相合。
        通俗暗合组：寅丑, 午亥, 卯申, 巳酉(丙辛合), 辰(戊癸)...
        """
        stems1 = self.hidden_stems.get(zhi1, [])
        stems2 = self.hidden_stems.get(zhi2, [])
        found = []
        for s1 in stems1:
            for s2 in stems2:
                # 检查 s1 和 s2 是否在天干五合里
                pair = sorted([s1, s2]) # 排序以便查找
                # 这里我们遍历 self.gan_rel_map["he"] 来匹配
                for he_pair in self.gan_rel_map["he"]:
                    if pair[0] == sorted(he_pair)[0] and pair[1] == sorted(he_pair)[1]:
                        found.append(f"{s1}{s2}合")
        
        if found:
            return f"暗合(藏干{','.join(found)})"
        return None

    def analyze_detailed_relations(self, pillars):
        """
        全面分析八字关系
        pillars: dict {'年':('辛','巳'), '月':('丁','酉'), ...}
        """
        messages = []
        
        # 提取天干和地支列表，带上位置标签以便区分
        gans = [] # [(位置, 天干), ...]
        zhis = [] # [(位置, 地支), ...]
        
        for loc, (g, z) in pillars.items():
            gans.append((loc, g))
            zhis.append((loc, z))

        # --- 1. 天干关系分析 ---
        # 使用 combinations 检查所有两两组合 (C(4,2) = 6对)
        for i in range(len(gans)):
            for j in range(i + 1, len(gans)):
                loc1, g1 = gans[i]
                loc2, g2 = gans[j]
                pair_set = frozenset([g1, g2])
                pair_sorted = tuple(sorted([g1, g2]))
                
                relation_found = False
                
                # 检查冲
                for chong_pair in self.gan_rel_map["chong"]:
                    if pair_sorted == tuple(sorted(chong_pair)):
                        messages.append(f"天干【{g1}{g2}】相冲 ({loc1}-{loc2})")
                        relation_found = True
                
                # 检查合
                for he_pair in self.gan_rel_map["he"]:
                    if pair_sorted == tuple(sorted(he_pair)):
                        hua = self.gan_he_hua.get(he_pair, "")
                        messages.append(f"天干【{g1}{g2}】五合化{hua} ({loc1}-{loc2})")
                        relation_found = True
                
                # 检查克 (如果没有冲合，再报克，避免信息冗余，或者都报)
                # 丁癸既冲也克，通常报冲即可。丁辛是克。
                if not relation_found:
                    ke_msg = self.check_wuxing_ke(g1, g2)
                    if ke_msg:
                        messages.append(f"天干【{ke_msg}】 ({loc1}-{loc2})")

        # --- 2. 地支关系分析 ---
        zhi_list = [z for _, z in zhis] # 纯地支列表 ['巳', '酉', '未', '申']
        
        # 2.1 两两关系 (六合, 六冲, 相害, 相破, 相刑, 暗合)
        for i in range(len(zhis)):
            for j in range(i + 1, len(zhis)):
                loc1, z1 = zhis[i]
                loc2, z2 = zhis[j]
                pair_set = frozenset([z1, z2])
                
                # 六合
                if pair_set in self.zhi_liu_he:
                    messages.append(f"地支【{z1}{z2}】六合{self.zhi_liu_he[pair_set]} ({loc1}-{loc2})")
                
                # 六冲
                if pair_set in self.zhi_liu_chong:
                    messages.append(f"地支【{z1}{z2}】相冲 ({loc1}-{loc2})")
                
                # 相害 (穿)
                if pair_set in self.zhi_hai:
                    messages.append(f"地支【{z1}{z2}】相害 ({loc1}-{loc2})")
                    
                # 相破
                if pair_set in self.zhi_po:
                    messages.append(f"地支【{z1}{z2}】相破 ({loc1}-{loc2})")
                
                # 暗合 (新加功能)
                an_he = self.check_an_he(z1, z2)
                if an_he:
                    # 过滤掉已经是六合的，避免重复描述，或者共存
                    if pair_set not in self.zhi_liu_he:
                        messages.append(f"地支【{z1}{z2}】{an_he} ({loc1}-{loc2})")

                # 自刑
                if z1 == z2 and z1 in self.zhi_xing["自刑"]:
                    messages.append(f"地支【{z1}{z1}】自刑 ({loc1}-{loc2})")

        # 2.2 复杂相刑 (涉及三个地支的循环刑)
        # 检查 寅巳申
        present_zhis = set(zhi_list)
        if {"寅", "巳", "申"}.issubset(present_zhis):
            messages.append("地支【寅巳申】三刑俱全 (无恩之刑)")
        elif "寅" in present_zhis and "巳" in present_zhis: messages.append("地支【寅巳】相刑")
        elif "巳" in present_zhis and "申" in present_zhis: messages.append("地支【巳申】相刑") # 这里会和六合重复，但命理上不同
        elif "申" in present_zhis and "寅" in present_zhis: messages.append("地支【寅申】相刑") # 也是冲

        # 检查 丑未戌
        if {"丑", "未", "戌"}.issubset(present_zhis):
            messages.append("地支【丑未戌】三刑俱全 (恃势之刑)")
        else:
            # 简单的两两刑在上面通常不单独报丑未(冲)，主要报未戌、丑戌
            pass 

        # 2.3 三合与拱合 (重点：你的巳酉、巳未需求)
        # 逻辑：检查每个“三合局”里的字在原局出现了几个
        for label, group in self.zhi_san_he.items():
            # group = ["巳", "酉", "丑"]
            count = 0
            found_items = []
            for item in group:
                if item in zhi_list:
                    count += 1
                    found_items.append(item)
            
            found_items = sorted(list(set(found_items)), key=group.index) # 按生旺墓排序
            
            if count == 3:
                messages.append(f"地支【{''.join(group)}】三合{label} (成局)")
            elif count == 2:
                # 检查是半合还是拱合
                s = "".join(found_items)
                if s == group[0]+group[1]: # 生+旺 (如 巳酉)
                    messages.append(f"地支【{s}】半合{label}")
                elif s == group[1]+group[2]: # 旺+墓 (如 酉丑)
                    messages.append(f"地支【{s}】半合{label}")
                elif s == group[0]+group[2]: # 生+墓 (如 巳丑) -> 拱
                    messages.append(f"地支【{s}】拱合{label} (缺中神)")

        # 2.4 三会与拱会 (重点：巳未拱会火)
        for label, group in self.zhi_san_hui.items():
            # group = ["巳", "午", "未"]
            present = [z for z in group if z in zhi_list]
            present = sorted(list(set(present)), key=group.index)
            
            if len(present) == 3:
                messages.append(f"地支【{''.join(group)}】三会{label} (一方之气)")
            elif len(present) == 2:
                s = "".join(present)
                # 比如 巳午，午未，或者 巳未(拱)
                if abs(group.index(present[0]) - group.index(present[1])) == 2:
                    # 索引差2，说明中间缺了一个，比如 巳(0) 和 未(2)，缺 午
                    messages.append(f"地支【{s}】拱会{label}")
                else:
                    # 简单的半会，力量较三合小，但也算党羽
                    messages.append(f"地支【{s}】半会{label} (同气)")

        return list(set(messages)) # 去重

    def generate_prompt(self, year, month, day, hour, minute, gender_str):
        # 1. 基础排盘
        solar = Solar.fromYmdHms(year, month, day, hour, minute, 0)
        lunar = solar.getLunar()
        bazi = lunar.getEightChar()
        gender_code = 1 if gender_str == "male" else 0
        bazi.setSect(1)

        year_gan, year_zhi = bazi.getYearGan(), bazi.getYearZhi()
        month_gan, month_zhi = bazi.getMonthGan(), bazi.getMonthZhi()
        day_gan, day_zhi = bazi.getDayGan(), bazi.getDayZhi()
        hour_gan, hour_zhi = bazi.getTimeGan(), bazi.getTimeZhi()

        pillars = {
            "年": (year_gan, year_zhi),
            "月": (month_gan, month_zhi),
            "日": (day_gan, day_zhi),
            "时": (hour_gan, hour_zhi)
        }

        relation_notes = self.analyze_detailed_relations(pillars)

        cang_gan_info = []
        for p_name, (_, zhi) in pillars.items():
            stems = self.hidden_stems.get(zhi, [])
            info = f"{p_name}支({zhi}): " + "/".join([f"{s}[{self.gan_wuxing[s]}]" for s in stems])
            cang_gan_info.append(info)

        all_chars = [c for p in pillars.values() for c in p]
        wuxing_list = [self.gan_wuxing.get(c, self.zhi_wuxing.get(c)) for c in all_chars]
        wuxing_count = {x: wuxing_list.count(x) for x in ["金", "木", "水", "火", "土"]}

        yun = bazi.getYun(gender_code)
        start_year = yun.getStartSolar().getYear()
        da_yun_str = []
        da_yun_list = yun.getDaYun()
        for i in range(1, 9):
            dy = da_yun_list[i]
            da_yun_str.append(f"({dy.getGanZhi()}, {dy.getStartAge()}-{dy.getStartAge()+9}岁, {dy.getStartYear()}-{dy.getStartYear()+9}年)")

        gan_relations = [r for r in relation_notes if "天干" in r]
        zhi_relations = [r for r in relation_notes if "地支" in r]
        newline = "\n"
        gan_relations_str = f"{newline}".join(gan_relations) if gan_relations else "无明显冲合"
        zhi_relations_str = f"{newline}".join(zhi_relations) if zhi_relations else "无明显关系"
        da_yun_sequence = "\n".join(da_yun_str)

        hour_str = f"{hour:02d}"
        minute_str = f"{minute:02d}"

        # ✅ NEW: 运行时“当前年份/流年”
        runtime = self.get_runtime_year_info()

        prompt = f"""
1. 基础信息
性别: {gender_str}
公历: {year}年{month}月{day}日{hour_str}:{minute_str}

2. 八字排盘
年柱： {year_gan}({self.gan_wuxing[year_gan]})|{year_zhi}({self.zhi_wuxing[year_zhi]}) 地支藏干：{self.hidden_stems[year_zhi]}
月柱： {month_gan}({self.gan_wuxing[month_gan]})|{month_zhi}({self.zhi_wuxing[month_zhi]}) 地支藏干：{self.hidden_stems[month_zhi]}
日柱： {day_gan}({self.gan_wuxing[day_gan]})|{day_zhi}({self.zhi_wuxing[day_zhi]}) 地支藏干：{self.hidden_stems[day_zhi]}
时柱： {hour_gan}({self.gan_wuxing[hour_gan]})|{hour_zhi}({self.zhi_wuxing[hour_zhi]}) 地支藏干：{self.hidden_stems[hour_zhi]}

3. 命局关系分析
五行统计: {wuxing_count} (显式, 不包括地支藏干以及合化五行)
天干关系:\n{gan_relations_str}
地支关系 (含刑冲合害破、拱会、暗合):\n{zhi_relations_str}

4. 大运
{da_yun_sequence}

5. 当前时间（以运行时系统时间为准）
当前公历时间: {runtime["current_solar_str"]}
当前流年(年柱干支): {runtime["liu_nian_ganzhi"]}
流年五行: 天干{runtime["liu_nian_gan"]}[{runtime["liu_nian_gan_wuxing"]}] / 地支{runtime["liu_nian_zhi"]}[{runtime["liu_nian_zhi_wuxing"]}]
        """
        return prompt
