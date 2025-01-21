import asyncio
import json
import unittest
from pathlib import Path
from typing import Annotated, List

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent, capture_run_messages
from unstructured.documents.elements import Element

from api.core.logger import log
from api.services.etl.chunking import ChunkingOptions, chunk_by_semantic_sections
from api.services.etl.cleaning import CleaningOptions, TextCleaner
from api.services.etl.extract import (
    DocumentExtractor,
    ExtractedDocument,
    ExtractionOptions,
)
from api.services.etl.partitioning import (
    PartitionHTMLConfig,
    PartitionPDFConfig,
    partition_html_from_url,
    partition_pdf_file,
)
from api.services.langchain.langchain_ollama import OllamaService
from api.services.pydantic_ai.vertexai_service.vertexai_service import VertexAIService
from api.utils.jsonify import Jsonify

SAMPLE_PDF = "static/assets/test_files/resmi-gazete.pdf"
SAMPLE_JSON = "static/assets/test_files/turk-ceza-kanunu.json"
SAMPLE_TXT = "static/assets/junk/sample.txt"
SAMPLE_PDF_URL = "https://www.resmigazete.gov.tr/eskiler/2025/01/20250102.pdf"
SAMPLE_HTML_URL = "https://www.meb.gov.tr/mevzuat/liste.php?ara=6"


class TurkCezaKanunMaddeleri(BaseModel):
    """Türk Ceza Kanununun içerisinde yer alan maddeler"""

    madde_no: int = Field(..., description="Kanun maddesinin numarası")
    madde_basligi: str = Field(..., description="Kanun maddesinin başlığı")
    madde_metni: str = Field(..., description="Kanun maddesinin metni")
    madde_ozeti: str = Field(..., description="Kanun maddesinin özeti")
    madde_amac: str = Field(..., description="Kanun maddesinin amacı")

    kitap_no: int = Field(
        ..., description="Kanun metni içerisinde kitap olarak ayrılmış kısmın numarası"
    )
    kitap_adi: str = Field(
        ..., description="Kanun metni içerisinde kitap olarak ayrılmış kısmın adı"
    )

    kisim_no: int = Field(
        ..., description="Kanun metni içerisinde kısım olarak ayrılmış kısmın numarası"
    )
    kisim_adi: str = Field(
        ..., description="Kanun metni içerisinde kısım olarak ayrılmış kısmın adı"
    )

    bolum_no: int = Field(
        ..., description="Kanun metni içerisinde bölüm olarak ayrılmış kısmın numarası"
    )
    bolum_adi: str = Field(
        ..., description="Kanun metni içerisinde bölüm olarak ayrılmış kısmın adı"
    )

    statu: str = Field(..., description="Maddenin güncel durumu")
    referanslar: List[str]

    category: str = Field(..., description="Maddenin kategorisi")
    tags: List[str] = Field(..., description="Maddenin etiketleri")

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


# Define the schema for the response
class TurkCezaKanunu(BaseModel):
    """Türk Ceza Kanunu"""

    kanun_adi: str = Field(..., description="Kanunun adı")
    kanun_numarasi: int = Field(..., description="Kanunun numarası")
    resmi_gazete_yayin_tarihi: str = Field(
        ..., description="Kanunun resmi gazete yayın tarihi"
    )
    maddeler: List[TurkCezaKanunMaddeleri] = Field(..., description="Kanunun maddeleri")
    category: str = Field(..., description="Kanunun kategorisi")
    tags: List[str] = Field(..., description="Kanunun etiketleri")

    class Config:
        json_schema_extra = {
            "example": {
                "kanun_adi": "Türk Ceza Kanunu",
                "kanun_numarasi": 1,
                "resmi_gazete_yayin_tarihi": "12/10/2004",
                "maddeler": [],
                "category": "Ceza Kanunu",
                "tags": ["ceza", "hukuk"],
            }
        }


class TestETL(unittest.TestCase):
    def setUp(self):
        self.file_path = Path(SAMPLE_PDF)
        self.cleaner = TextCleaner(options=CleaningOptions())
        # self.file_content: List[Element] = partition_pdf_file(PartitionPDFConfig(filename=SAMPLE_PDF))
        self.file_content: List[Element] = partition_html_from_url(
            PartitionHTMLConfig(url=SAMPLE_PDF_URL)
        )

        self.print_elements(self.file_content)

    def print_elements(self, elements: List[Element]):
        for element in elements:
            log.info(f"\n\n---------")
            el_type = element.metadata.to_dict().get("type")
            if el_type == "Image" or el_type == "Table":
                print("Table: ", element.metadata.text_as_htmlt)
            log.info(f"\n\n{element.to_dict()}")
            log.info(f"\n\n{element.metadata.to_dict()}")

    @pytest.mark.skip(reason="Later check")
    def generate_prompt(self, unstructured_text: str):
        schema = json.dumps(
            TurkCezaKanunu.model_json_schema(), indent=2, ensure_ascii=False
        )
        return f"""
		Aşağıdaki bilgi tabanını kullanarak sağlanan bağlamdan yapılandırılmış verileri çıkarın. 
		Çıktı tam olarak bu JSON şemasına uymalıdır:

		{schema}

		## JSON Düzeltme Kuralları
		- Tüm özellik adlarını ve dize değerlerini çift tırnak işareti (`"`) içine alın.
		- Nesnelerdeki veya dizilerdeki son virgülleri kaldırın.
		- Anahtarlar veya değerler etrafındaki tek tırnak işaretlerini çift tırnak işaretiyle değiştirin.
		- Orijinal JSON'un yapısının ve değerlerinin korunduğundan emin olun.
		- Düzeltilen JSON tek çıktı olmalıdır.

		# Çıktı Biçimi
		Yukarıdaki şemaya göre biçimlendirilmiş bir JSON nesnesi sağlayın.

		# Notlar
		Ham veri biçimlerini korumada ve bilgi tabanına göre doğru eşlemeleri uygulamada ayrıntılara dikkat edin.
		Yapılandırılmış data üreteceğin metin şu: {unstructured_text}"""

    @pytest.mark.skip(reason="Later check")
    def test_extract_document(self):
        """Test document extraction."""
        assert self.file_content is not None
        assert len(self.file_content) > 0
        assert all(isinstance(e, Element) for e in self.file_content)

    def test_extract_from_html(self):
        """Test document extraction from url."""
        log.info(f"\n\nTest document extraction from url.")
        file_content: List[Element] = partition_html_from_url(
            PartitionHTMLConfig(url=SAMPLE_PDF_URL)
        )
        self.print_elements(file_content)
        assert file_content is not None
        assert len(file_content) > 0
        assert all(isinstance(e, Element) for e in file_content)

    @pytest.mark.skip(reason="Later check")
    def test_clean_document(self):
        """Test document cleaning."""
        cleaned_elements = self.cleaner.clean_elements(self.file_content)
        assert cleaned_elements is not None
        assert len(cleaned_elements) > 0
        assert all(isinstance(e, Element) for e in cleaned_elements)

    @pytest.mark.skip(reason="Not implemented")
    def _test_semantic_relations_chunking(self):
        """Test semantic chunking."""
        cleaned_elements = self.cleaner.clean_elements(self.file_content)
        chunking_options = ChunkingOptions(
            max_characters=8000, combine_under_n_chars=500, chunking_strategy="by_title"
        )
        chunks = chunk_by_semantic_sections(cleaned_elements, chunking_options)
        assert chunks is not None
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text is not None
            assert len(chunk.text) > 0
            assert chunk.metadata is not None
            if chunk.orig_elements:
                assert all(isinstance(e, Element) for e in chunk.orig_elements)

    @pytest.mark.skip(reason="Not implemented")
    def test_generate_partitions_by_prompt(self):
        # asyncio.run(self._test_generate_partitions_by_prompt())
        # asyncio.run(self._test_generate_partitions_by_gemini())
        pass

    def _test_generate_partitions_by_gemini(self):
        try:
            cleaned_elements = self.cleaner.clean_elements(elements=self.file_content)
            cleaned_text = "\n\n".join([e.text for e in cleaned_elements])
            cleaned_text = self.cleaner.clean_text(text=cleaned_text)
            log.info(f"Cleaned text: {cleaned_text}")
        except Exception as e:
            log.error(f"Error in test data cleaning: {e}")
            raise e
        try:
            vertexai_service = VertexAIService()
            agent: Agent[None, TurkCezaKanunu] = vertexai_service.get_agent(
                result_type=TurkCezaKanunu,
                name="turk_ceza_kanunu_agent",
                retries=1,
                end_strategy="exhaustive",
                defer_model_check=True,
                system_prompt="""
                    Türk Ceza Kanunu'nun metni hakkında bir fikir sahibi olduktan sonra 
                    teker teker tüm maddeleri, bölümleri inceleyip, sana verilen JSON şemasına göre 
                    Bir yapılandırılmış JSONdata üretmelisin:
                """,
            )

            with capture_run_messages() as messages:
                try:
                    result = agent.run_sync(cleaned_text)
                    log.info(f"\n\n\nResult.data: {result.data}")
                except Exception:
                    log.error(f"\n\n\nError in agent run: {messages}")
                    raise
        except Exception as e:
            log.error(f"\n\n\nError in agent run: {e}")
            raise e
        try:
            json.dump(
                obj=result.data.model_dump(),
                fp=open(str(Path(SAMPLE_TXT)), "w"),
                indent=2,
                ensure_ascii=False,
            )
        except Exception as e:
            log.error(f"\n\n\nError in save data: {e}")
            raise e

    @pytest.mark.asyncio
    async def _test_generate_partitions_by_prompt(self):
        """Test generating partitions by prompt."""
        try:
            cleaned_elements = self.cleaner.clean_elements(elements=self.file_content)
            cleaned_text = "\n\n".join([e.text for e in cleaned_elements])
            cleaned_text = self.cleaner.clean_text(text=cleaned_text)
            log.info(
                f"test_generate_partitions_by_prompt: Cleaned text: {cleaned_text}"
            )

            # Take first chunk for testing
            chat_response = await OllamaService().generate_structured_output(
                output_schema=TurkCezaKanunu,
                message=f"""Tüm kanunun metni hakkında bir fikir sahibi olduktan sonra 
                    teker teker tüm maddeleri, bölümleri inceleyip, sana verilen JSON şemasına göre 
                    JSON structured_output'u üretmelisin:
                    Türk Ceza Kanunu'nun metni:
                        '''
                        {cleaned_text}
                        '''
                    """,
            )
            log.info(f"Chat response: {chat_response}")
            json.dump(
                obj=chat_response.model_dump(),
                fp=open(str(Path(SAMPLE_JSON)), "w"),
                indent=2,
                ensure_ascii=False,
            )

            log.info(f"Chat response: {chat_response}")
            assert chat_response is not None

        except Exception as e:
            log.error(f"Error in test: {e}")
            raise e
