"""
Local demo script: test the whole agent flow directly from the terminal,
without needing to start the FastAPI server.

Usage:
    python demo_cli.py
"""
from app.core.rag_engine import rag_engine
from app.core.agent_graph import run_agent


def main():
    print("=== 步驟 1：匯入範例文件到知識庫 ===")
    n = rag_engine.ingest_files(["data/sample_docs/employee_handbook_sample.txt"])
    print(f"寫入 {n} 個切片\n")

    questions = [
        "員工一年可以請幾天特休？",          # Should hit the knowledge base
        "遠端工作一個月最多幾天？",          # Should hit the knowledge base
        "2026 年台灣颱風假的最新規定是什麼？",  # Not in the knowledge base, should trigger a web search
    ]

    for q in questions:
        print(f"=== 問題：{q} ===")
        result = run_agent(q)
        print(f"答案：{result['answer']}")
        print(f"需要人工複核：{result['needs_human_review']}")
        print(f"執行步數：{result['step_count']}\n")


if __name__ == "__main__":
    main()
