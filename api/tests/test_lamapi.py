import httpx
import asyncio
import time  # Import time for measuring duration


async def test_lamapi_diagnostic():
    targets = ["http://127.0.0.1:8000"]
    params = {
        "name": "McDonald Whopper hamburger",
        "kg": "wikidata",
        "limit": "10",
        "token": "lamapi_demo_2023"
    }

    for base_url in targets:
        url = f"{base_url}/lookup/entity-retrieval"
        print(f"🔍 Testing: {url}")

        try:
            async with httpx.AsyncClient(timeout=360.0) as client:
                # --- START TIMER ---
                start_time = time.perf_counter()

                response = await client.get(url, params=params)

                # --- END TIMER ---
                end_time = time.perf_counter()
                duration = end_time - start_time

                if response.status_code == 200:
                    print(f"✅ Status: 200")
                    print(f"⏱️ Time taken: {duration:.2f} seconds")  # Prints time to 2 decimal places

                    data = response.json()
                    print(f"📄 Found {len(data)} entities.")
                    if data:
                        print(f"🍎 Top Result: {data[0].get('name')} ({data[0].get('id')})")
                    return
                else:
                    print(f"⚠️ Status {response.status_code} in {duration:.2f}s")

        except httpx.TimeoutException:
            print(f"❌ Timeout after 360s")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_lamapi_diagnostic())