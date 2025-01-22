from enum import Enum
from typing import Annotated, List

from pydantic import BaseModel, Field
from pydantic_extra_types.language_code import LanguageAlpha2


class YasalBelgeTipi(Enum):
    KANUN = "Kanun"
    YÖNETMELİK = "Yönetmelik"
    CB_KARARNAME = "Cumhurbaşkanliği Kararnameleri̇"
    CB_VE_BK_YONETMELİK = "Cumhurbaşkanliği Ve Bakanlar Kurulu Yönetmeli̇kleri̇"
    CB_KARAR = "Cumhurbaşkanı Kararlari̇"
    CB_GENELGE = "Cumhurbaşkanliği Genelgeleri̇"
    KHK = "Kanun Hükmünde Kararnameler"
    TUZUK = "Tüzükler"
    KURUM_KURULUS_VE_UNIVERSITE_YONETMELIK = (
        "Kurum, Kuruluş Ve Üni̇versi̇te Yönetmeli̇kleri̇"
    )
    TEBLIG = "Teblig"
    RESMI_GAZETE = "Resmi Gazete"
    AM_BIREYSEL_BASVURU_KARAR = "Anayasa Bireysel Basvuru Kararlari̇"
    AM_KARAR = "Anayasa Kararlari̇"
    YARGITAY_EMSAL_KARAR = "Yargıtay Emsal Kararlari̇"
    MEVZUAT = "Mevzuat"


class YasalBelgeMaddesi(BaseModel):
    """Türkiye Cumhuriyeti Yasal Belgesinin içerisinde yer alan tekil maddeler"""

    madde_no: int = Field(..., description="Yasal maddenin numarası")

    madde_basligi: str = Field(..., description="Yasal maddenin ait olduğu başlık")

    madde_metni: str = Field(..., description="Yasal maddenin metni")

    madde_ozeti: str = Field(..., description="Yasal maddenin subjectif özeti")

    madde_amac: str = Field(..., description="Yasal maddenin belirtilmiş amacı")

    kitap_no: int = Field(
        ..., description="Yasal metin içerisinde kitap olarak ayrılmış kısmın numarası"
    )
    kitap_adi: str = Field(
        ..., description="Yasal metin içerisinde kitap olarak ayrılmış kısmın adı"
    )

    kisim_no: int = Field(
        ..., description="Yasal metin içerisinde kısım olarak ayrılmış kısmın numarası"
    )
    kisim_adi: str = Field(
        ..., description="Yasal metin içerisinde kısım olarak ayrılmış kısmın adı"
    )

    bolum_no: int = Field(
        ..., description="Yasal metin içerisinde bölüm olarak ayrılmış kısmın numarası"
    )
    bolum_adi: str = Field(
        ..., description="Yasal metin içerisinde bölüm olarak ayrılmış kısmın adı"
    )

    statu: str = Field(..., description="Maddenin güncel durumu")
    referanslar: List[str]

    category: str = Field(..., description="Yasal maddenin kategorisi")
    tags: List[str] = Field(..., description="Yasal maddenin etiketleri")

    class Config:
        json_schema_extra = {
            "example": {
                "madde_no": 1,
                "madde_basligi": "Örnek Başlık",
                "madde_metni": "Örnek metin",
                "madde_ozeti": "Özet",
                "madde_amac": "Amaç",
                "kitap_no": 1,
                "kitap_adi": "Birinci Kitap",
                "kisim_no": 1,
                "kisim_adi": "Birinci Kısım",
                "bolum_no": 1,
                "bolum_adi": "Birinci Bölüm",
                "statu": "Yürürlükte",
                "referanslar": ["Madde 2", "Madde 3"],
                "category": "Genel Hükümler",
                "tags": ["ceza", "hukuk"],
            }
        }


class YasalBelge(BaseModel):
    """Türkiye Cumhuriyeti Yasal Belgesi"""

    belge_adi: str = Field(..., description="Yasal belgenin adı")
    belge_dili: LanguageAlpha2 = Field(
        default=LanguageAlpha2("tr"), description="Yasal belgenin dili"
    )
    belge_numarasi: int = Field(..., description="Yasal belge numarası")
    belge_tipi: YasalBelgeTipi = Field(..., description="Yasal belgenin tipi")
    resmi_gazete_yayin_tarihi: str = Field(
        ..., description="Yasal belgenin resmi gazete yayın tarihi"
    )
    maddeler: List[YasalBelgeMaddesi] = Field(
        ..., description="Yasal belgenin içerisinde yer alan maddelerin tümü"
    )
    category: str = Field(..., description="Yasal belge kategorisi")
    tags: List[str] = Field(..., description="Yasal belge etiketleri")

    class Config:
        json_schema_extra = {
            "example": {
                "belge_adi": "Türk Ceza Kanunu",
                "belge_dili": "tr",
                "belge_numarasi": 1,
                "belge_tipi": "Kanun",
                "resmi_gazete_yayin_tarihi": "12/10/2004",
                "maddeler": [],
                "category": "Ceza Kanunu",
                "tags": ["ceza", "hukuk"],
            }
        }
