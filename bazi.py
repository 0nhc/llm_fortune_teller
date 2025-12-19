from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from lunar_python import Solar

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    _HAS_ZONEINFO = True
except Exception:  # pragma: no cover
    _HAS_ZONEINFO = False


Gan = str
Zhi = str
Pillars = Dict[str, Tuple[Gan, Zhi]]


class BaZiAutomation:
    """
    Utilities for generating BaZi (Four Pillars) structured signals and a formatted prompt.

    The implementation focuses on:
      - Five Elements (Wu Xing) mappings and relations
      - Stem/Branch relationship detection (he/chong/xing/hai/po, etc.)
      - Runtime "current year / Liu Nian" computation in a specified timezone
    """

    def __init__(self) -> None:
        # Heavenly Stems (天干) -> Wu Xing
        self.gan_wuxing: Dict[Gan, str] = {
            "甲": "木", "乙": "木",
            "丙": "火", "丁": "火",
            "戊": "土", "己": "土",
            "庚": "金", "辛": "金",
            "壬": "水", "癸": "水",
        }

        # Earthly Branches (地支) -> Wu Xing
        self.zhi_wuxing: Dict[Zhi, str] = {
            "子": "水", "丑": "土",
            "寅": "木", "卯": "木",
            "辰": "土", "巳": "火",
            "午": "火", "未": "土",
            "申": "金", "酉": "金",
            "戌": "土", "亥": "水",
        }

        # Wu Xing generation (生) and overcoming (克) relations
        self.wuxing_relation = {
            "生": [("木", "火"), ("火", "土"), ("土", "金"), ("金", "水"), ("水", "木")],
            "克": [("木", "土"), ("土", "水"), ("水", "火"), ("火", "金"), ("金", "木")],
        }

        # Hidden stems (地支藏干)
        self.hidden_stems: Dict[Zhi, List[Gan]] = {
            "子": ["癸"], "丑": ["己", "癸", "辛"], "寅": ["甲", "丙", "戊"], "卯": ["乙"],
            "辰": ["戊", "乙", "癸"], "巳": ["丙", "戊", "庚"], "午": ["丁", "己"], "未": ["己", "丁", "乙"],
            "申": ["庚", "壬", "戊"], "酉": ["辛"], "戌": ["戊", "辛", "丁"], "亥": ["壬", "甲"],
        }

        # Heavenly stem relationships
        self.gan_rel_map = {
            "chong": [("甲", "庚"), ("乙", "辛"), ("丙", "壬"), ("丁", "癸")],  # 冲
            "he": [("甲", "己"), ("乙", "庚"), ("丙", "辛"), ("丁", "壬"), ("戊", "癸")],  # 五合
        }

        # Five combinations -> transformation element (simplified hinting)
        self.gan_he_hua: Dict[Tuple[Gan, Gan], str] = {
            ("甲", "己"): "土",
            ("乙", "庚"): "金",
            ("丙", "辛"): "水",
            ("丁", "壬"): "木",
            ("戊", "癸"): "火",
        }

        # Precompute a canonical set for quick "he" checks in hidden stem matching
        self._gan_he_canon = {tuple(sorted(p)) for p in self.gan_rel_map["he"]}

        # Earthly branch relationships
        self.zhi_liu_he = {
            frozenset(["子", "丑"]): "化土",
            frozenset(["寅", "亥"]): "化木",
            frozenset(["卯", "戌"]): "化火",
            frozenset(["辰", "酉"]): "化金",
            frozenset(["巳", "申"]): "化水",
            frozenset(["午", "未"]): "化土/火",
        }

        self.zhi_liu_chong = [
            frozenset(["子", "午"]),
            frozenset(["丑", "未"]),
            frozenset(["寅", "申"]),
            frozenset(["卯", "酉"]),
            frozenset(["辰", "戌"]),
            frozenset(["巳", "亥"]),
        ]

        # San He (三合) groups
        self.zhi_san_he = {
            "水局": ["申", "子", "辰"],
            "木局": ["亥", "卯", "未"],
            "火局": ["寅", "午", "戌"],
            "金局": ["巳", "酉", "丑"],
        }

        # San Hui (三会) groups
        self.zhi_san_hui = {
            "水会": ["亥", "子", "丑"],
            "木会": ["寅", "卯", "辰"],
            "火会": ["巳", "午", "未"],
            "金会": ["申", "酉", "戌"],
        }

        # Xing (刑)
        self.zhi_xing = {
            "无礼刑": [("子", "卯")],
            "恃势刑": [("寅", "巳"), ("巳", "申"), ("申", "寅")],
            "丑未戌刑": [("丑", "未"), ("未", "戌"), ("戌", "丑")],
            "自刑": ["辰", "午", "酉", "亥"],
        }

        # Hai (害/穿)
        self.zhi_hai = [
            frozenset(["子", "未"]),
            frozenset(["丑", "午"]),
            frozenset(["寅", "巳"]),
            frozenset(["卯", "辰"]),
            frozenset(["申", "亥"]),
            frozenset(["酉", "戌"]),
        ]

        # Po (破)
        self.zhi_po = [
            frozenset(["子", "酉"]),
            frozenset(["午", "卯"]),
            frozenset(["巳", "申"]),
            frozenset(["寅", "亥"]),
            frozenset(["辰", "丑"]),
            frozenset(["戌", "未"]),
        ]

    def _now_in_tz(self, tz_name: str) -> datetime:
        """Return timezone-aware current datetime in the given timezone."""
        if _HAS_ZONEINFO:
            return datetime.now(ZoneInfo(tz_name))

        # Fallback for older Python: pip install pytz
        import pytz  # type: ignore
        return datetime.now(pytz.timezone(tz_name))

    def _ensure_tz(self, dt: datetime, tz_name: str) -> datetime:
        """Ensure a datetime is timezone-aware; if naive, attach tz_name."""
        if dt.tzinfo is not None:
            return dt

        if _HAS_ZONEINFO:
            return dt.replace(tzinfo=ZoneInfo(tz_name))

        import pytz  # type: ignore
        return dt.replace(tzinfo=pytz.timezone(tz_name))

    def get_runtime_year_info(
        self,
        as_of: Optional[datetime] = None,
        tz_name: str = "Asia/Shanghai",
    ) -> Dict[str, str | int | datetime]:
        """
        Compute runtime (current) Gregorian time and Liu Nian (流年, year pillar) in a given timezone.

        Notes:
          - The year pillar commonly uses LiChun (立春) as the boundary in BaZi practice.
          - This method follows lunar_python's EightChar year pillar convention.

        Args:
            as_of: Optional datetime override for reproducibility/testing.
                   If naive, tz_name is attached.
            tz_name: IANA timezone name (default: Asia/Shanghai).

        Returns:
            A dict containing:
              - now (datetime): timezone-aware current datetime
              - tz_name (str)
              - current_gregorian_year (int)
              - current_solar_str (str)
              - current_lunar_str (str)
              - liu_nian_ganzhi (str)
              - liu_nian_gan / liu_nian_zhi (str)
              - liu_nian_gan_wuxing / liu_nian_zhi_wuxing (str)
        """
        now = self._ensure_tz(as_of, tz_name) if as_of is not None else self._now_in_tz(tz_name)

        solar_now = Solar.fromYmdHms(
            now.year, now.month, now.day,
            now.hour, now.minute, now.second,
        )
        lunar_now = solar_now.getLunar()
        ec_now = lunar_now.getEightChar()
        ec_now.setSect(1)

        ly_gan: Gan = ec_now.getYearGan()
        ly_zhi: Zhi = ec_now.getYearZhi()

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

    def check_wuxing_ke(self, gan1: Gan, gan2: Gan) -> Optional[str]:
        """Return a short message if two stems form a Wu Xing overcoming (克) relation."""
        w1, w2 = self.gan_wuxing[gan1], self.gan_wuxing[gan2]
        if (w1, w2) in self.wuxing_relation["克"]:
            return f"{gan1}{gan2}相克"
        if (w2, w1) in self.wuxing_relation["克"]:
            return f"{gan2}{gan1}相克"
        return None

    def check_an_he(self, zhi1: Zhi, zhi2: Zhi) -> Optional[str]:
        """
        Check "hidden combination" (暗合) between two branches via their hidden stems.

        We detect whether any hidden stem pair forms one of the Heavenly Stem Five Combinations (天干五合).
        This is a simplified heuristic used as a hint, not a full-rule adjudication.
        """
        stems1 = self.hidden_stems.get(zhi1, [])
        stems2 = self.hidden_stems.get(zhi2, [])

        found: List[str] = []
        for s1 in stems1:
            for s2 in stems2:
                if tuple(sorted([s1, s2])) in self._gan_he_canon:
                    found.append(f"{s1}{s2}合")

        if found:
            return f"暗合(藏干{','.join(found)})"
        return None

    def analyze_detailed_relations(self, pillars: Pillars) -> List[str]:
        """
        Analyze inter-pillar relationships for a BaZi chart.

        Args:
            pillars: dict like {'年':('辛','巳'), '月':('丁','酉'), '日':('甲','子'), '时':('丙','午')}

        Returns:
            A de-duplicated list of relationship notes (Chinese labels kept for compactness).
        """
        messages: List[str] = []

        gans: List[Tuple[str, Gan]] = []
        zhis: List[Tuple[str, Zhi]] = []
        for loc, (g, z) in pillars.items():
            gans.append((loc, g))
            zhis.append((loc, z))

        # 1) Heavenly stem relations (pairwise)
        for i in range(len(gans)):
            for j in range(i + 1, len(gans)):
                loc1, g1 = gans[i]
                loc2, g2 = gans[j]
                pair_sorted = tuple(sorted([g1, g2]))

                relation_found = False

                # Chong (冲)
                for chong_pair in self.gan_rel_map["chong"]:
                    if pair_sorted == tuple(sorted(chong_pair)):
                        messages.append(f"天干相冲 ({loc1}-{loc2})")
                        relation_found = True

                # He (合)
                for he_pair in self.gan_rel_map["he"]:
                    if pair_sorted == tuple(sorted(he_pair)):
                        hua = self.gan_he_hua.get(tuple(he_pair), "")
                        messages.append(f"天干五合化{hua} ({loc1}-{loc2})")
                        relation_found = True

                # Ke (克) as a fallback hint when no chong/he found
                if not relation_found:
                    ke_msg = self.check_wuxing_ke(g1, g2)
                    if ke_msg:
                        messages.append(f"天干 ({loc1}-{loc2})")

        # 2) Earthly branch relations
        zhi_list: List[Zhi] = [z for _, z in zhis]

        # 2.1 Pairwise relations (六合/六冲/害/破/暗合/自刑)
        for i in range(len(zhis)):
            for j in range(i + 1, len(zhis)):
                loc1, z1 = zhis[i]
                loc2, z2 = zhis[j]
                pair_set = frozenset([z1, z2])

                if pair_set in self.zhi_liu_he:
                    messages.append(f"地支六合{self.zhi_liu_he[pair_set]} ({loc1}-{loc2})")

                if pair_set in self.zhi_liu_chong:
                    messages.append(f"地支相冲 ({loc1}-{loc2})")

                if pair_set in self.zhi_hai:
                    messages.append(f"地支相害 ({loc1}-{loc2})")

                if pair_set in self.zhi_po:
                    messages.append(f"地支相破 ({loc1}-{loc2})")

                an_he = self.check_an_he(z1, z2)
                if an_he and pair_set not in self.zhi_liu_he:
                    messages.append(f"地支{an_he} ({loc1}-{loc2})")

                if z1 == z2 and z1 in self.zhi_xing["自刑"]:
                    messages.append(f"地支自刑 ({loc1}-{loc2})")

        # 2.2 Multi-branch xing patterns (simplified reporting)
        present_zhis = set(zhi_list)
        if {"寅", "巳", "申"}.issubset(present_zhis):
            messages.append("地支【寅巳申】三刑俱全 (无恩之刑)")
        elif "寅" in present_zhis and "巳" in present_zhis:
            messages.append("地支【寅巳】相刑")
        elif "巳" in present_zhis and "申" in present_zhis:
            messages.append("地支【巳申】相刑")
        elif "申" in present_zhis and "寅" in present_zhis:
            messages.append("地支【寅申】相刑")

        if {"丑", "未", "戌"}.issubset(present_zhis):
            messages.append("地支【丑未戌】三刑俱全 (恃势之刑)")

        # 2.3 San He (三合) and partial forms
        for label, group in self.zhi_san_he.items():
            found_items = [item for item in group if item in zhi_list]
            found_items = sorted(list(set(found_items)), key=group.index)
            count = len(found_items)

            if count == 3:
                messages.append(f"地支三合{label} (成局)")
            elif count == 2:
                s = "".join(found_items)
                if s == group[0] + group[1]:
                    messages.append(f"地支半合{label}")
                elif s == group[1] + group[2]:
                    messages.append(f"地支半合{label}")
                elif s == group[0] + group[2]:
                    messages.append(f"地支拱合{label} (缺中神)")

        # 2.4 San Hui (三会) and partial forms
        for label, group in self.zhi_san_hui.items():
            present = [z for z in group if z in zhi_list]
            present = sorted(list(set(present)), key=group.index)

            if len(present) == 3:
                messages.append(f"地支三会{label} (一方之气)")
            elif len(present) == 2:
                s = "".join(present)
                if abs(group.index(present[0]) - group.index(present[1])) == 2:
                    messages.append(f"地支拱会{label}")
                else:
                    messages.append(f"地支半会{label} (同气)")

        # De-duplicate while keeping output stable-ish
        return sorted(set(messages))

    def generate_prompt(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        gender_str: str,
        runtime_tz: str = "Asia/Shanghai",
    ) -> str:
        """
        Build a formatted prompt containing BaZi pillars, relationships, DaYun, and runtime Liu Nian.

        Args:
            year, month, day, hour, minute: Gregorian birth datetime components.
            gender_str: 'male' or 'female' (case-insensitive).
            runtime_tz: Timezone used for the runtime "current time / Liu Nian" block.

        Returns:
            A multi-section prompt string (Chinese section titles retained).
        """
        solar = Solar.fromYmdHms(year, month, day, hour, minute, 0)
        lunar = solar.getLunar()
        bazi = lunar.getEightChar()

        gender_norm = gender_str.strip().lower()
        gender_code = 1 if gender_norm == "male" else 0
        bazi.setSect(1)

        year_gan, year_zhi = bazi.getYearGan(), bazi.getYearZhi()
        month_gan, month_zhi = bazi.getMonthGan(), bazi.getMonthZhi()
        day_gan, day_zhi = bazi.getDayGan(), bazi.getDayZhi()
        hour_gan, hour_zhi = bazi.getTimeGan(), bazi.getTimeZhi()

        pillars: Pillars = {
            "年": (year_gan, year_zhi),
            "月": (month_gan, month_zhi),
            "日": (day_gan, day_zhi),
            "时": (hour_gan, hour_zhi),
        }

        relation_notes = self.analyze_detailed_relations(pillars)

        # Visible Wu Xing counts (stems + branches only; no hidden stems, no transformations)
        all_chars = [c for p in pillars.values() for c in p]
        wuxing_list = [self.gan_wuxing.get(c, self.zhi_wuxing.get(c)) for c in all_chars]
        wuxing_count = {x: wuxing_list.count(x) for x in ["金", "木", "水", "火", "土"]}

        # DaYun
        yun = bazi.getYun(gender_code)
        da_yun_list = yun.getDaYun()
        da_yun_str: List[str] = []
        for i in range(1, 9):
            dy = da_yun_list[i]
            da_yun_str.append(
                f"({dy.getGanZhi()}, {dy.getStartAge()}-{dy.getStartAge()+9}岁, {dy.getStartYear()}-{dy.getStartYear()+9}年)"
            )
        da_yun_sequence = "\n".join(da_yun_str)

        gan_relations = [r for r in relation_notes if "天干" in r]
        zhi_relations = [r for r in relation_notes if "地支" in r]
        gan_relations_str = "\n".join(gan_relations) if gan_relations else "无明显冲合"
        zhi_relations_str = "\n".join(zhi_relations) if zhi_relations else "无明显关系"

        hour_str = f"{hour:02d}"
        minute_str = f"{minute:02d}"

        runtime = self.get_runtime_year_info(tz_name=runtime_tz)

        prompt = f"""
1. 基础信息
性别: {gender_norm}
公历: {year}年{month}月{day}日{hour_str}:{minute_str}

2. 八字排盘
年柱： {year_gan}({self.gan_wuxing[year_gan]})|{year_zhi}({self.zhi_wuxing[year_zhi]}) 地支藏干：{self.hidden_stems[year_zhi]}
月柱： {month_gan}({self.gan_wuxing[month_gan]})|{month_zhi}({self.zhi_wuxing[month_zhi]}) 地支藏干：{self.hidden_stems[month_zhi]}
日柱： {day_gan}({self.gan_wuxing[day_gan]})|{day_zhi}({self.zhi_wuxing[day_zhi]}) 地支藏干：{self.hidden_stems[day_zhi]}
时柱： {hour_gan}({self.gan_wuxing[hour_gan]})|{hour_zhi}({self.zhi_wuxing[hour_zhi]}) 地支藏干：{self.hidden_stems[hour_zhi]}

3. 命局关系分析
五行统计: {wuxing_count} (显式, 不包括地支藏干以及合化五行)
天干关系:
{gan_relations_str}
地支关系 (含刑冲合害破、拱会、暗合):
{zhi_relations_str}

4. 大运
{da_yun_sequence}

5. 当前时间（以运行时系统时间为准）
当前公历时间: {runtime["current_solar_str"]}
当前流年(年柱干支): {runtime["liu_nian_ganzhi"]}
流年五行: 天干{runtime["liu_nian_gan"]}[{runtime["liu_nian_gan_wuxing"]}] / 地支{runtime["liu_nian_zhi"]}[{runtime["liu_nian_zhi_wuxing"]}]
""".strip(
            "\n"
        )

        return prompt
