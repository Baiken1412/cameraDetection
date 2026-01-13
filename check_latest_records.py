"""
检查最新插入的轨迹记录
"""
import pymysql
from datetime import datetime, timedelta
import config_loader as config

def check_latest_records():
    """查询最近插入的轨迹记录"""
    try:
        # 连接数据库
        conn = pymysql.connect(
            host=config.DATABASE_CONFIG['host'],
            port=config.DATABASE_CONFIG['port'],
            user=config.DATABASE_CONFIG['user'],
            password=config.DATABASE_CONFIG['password'],
            database=config.DATABASE_CONFIG['database'],
            charset=config.DATABASE_CONFIG['charset'],
            cursorclass=pymysql.cursors.DictCursor
        )

        with conn.cursor() as cursor:
            # 查询最近10条记录的所有字段
            sql = """
                SELECT
                    id, pssj, jssj, pstp, qyid, qymc, sxtmx, spdz, spsc,
                    jqzt, bzzt, bzsj, xwyy, ryxm, wlry, rysl, jscs,
                    DATE_FORMAT(pssj, '%Y-%m-%d %H:%i:%s') as pssj_formatted
                FROM app_track
                ORDER BY id DESC
                LIMIT 10
            """
            cursor.execute(sql)
            records = cursor.fetchall()

            print("=" * 100)
            print(f"最近10条轨迹记录 (总共 {len(records)} 条):")
            print("=" * 100)

            for i, record in enumerate(records, 1):
                print(f"\n记录 #{i}:")
                print(f"  ID: {record['id']}")
                print(f"  拍摄时间 (pssj): {record['pssj_formatted']}")
                print(f"  结束时间 (jssj): {record['jssj']}")
                print(f"  图片路径 (pstp): {record['pstp']}")
                print(f"  区域ID (qyid): {record['qyid']}")
                print(f"  区域名称 (qymc): {record['qymc']}")
                print(f"  摄像头名称 (sxtmx): {record['sxtmx']}")
                print(f"  视频地址 (spdz): {record['spdz']}")
                print(f"  视频时长 (spsc): {record['spsc']}")
                print(f"  警情状态 (jqzt): {record['jqzt']}")
                print(f"  标注状态 (bzzt): {record['bzzt']} {'[OK]' if record['bzzt'] is not None else '[NULL]'}")
                print(f"  标注时间 (bzsj): {record['bzsj']}")
                print(f"  行为原因 (xwyy): {record['xwyy']}")
                print(f"  人员姓名 (ryxm): {record['ryxm']}")
                print(f"  外来人员 (wlry): {record['wlry']}")
                print(f"  人员数量 (rysl): {record['rysl']}")
                print(f"  检测次数 (jscs): {record['jscs']}")

                # 检查可能导致前端不显示的问题
                issues = []
                if record['bzzt'] is None:
                    issues.append("[X] bzzt 为 NULL")
                if record['pssj'] is None:
                    issues.append("[X] pssj 为 NULL")
                if record['qymc'] is None or record['qymc'] == '':
                    issues.append("[!] qymc 为空")

                if issues:
                    print(f"\n  [!] 可能的问题: {', '.join(issues)}")
                else:
                    print(f"\n  [OK] 字段完整，应该能在前端显示")

            # 统计今天的记录数
            print("\n" + "=" * 100)
            today = datetime.now().strftime('%Y-%m-%d')
            sql_today = """
                SELECT COUNT(*) as count
                FROM app_track
                WHERE DATE_FORMAT(pssj, '%Y-%m-%d') = %s
            """
            cursor.execute(sql_today, (today,))
            today_count = cursor.fetchone()['count']
            print(f"今天 ({today}) 的轨迹记录总数: {today_count}")

            # 统计不同bzzt状态的记录数
            sql_bzzt = """
                SELECT
                    bzzt,
                    COUNT(*) as count
                FROM app_track
                WHERE DATE_FORMAT(pssj, '%Y-%m-%d') = %s
                GROUP BY bzzt
            """
            cursor.execute(sql_bzzt, (today,))
            bzzt_stats = cursor.fetchall()
            print(f"\n今天的标注状态统计:")
            for stat in bzzt_stats:
                bzzt_label = "待标注" if stat['bzzt'] == '0' else ("已标注" if stat['bzzt'] == '1' else f"未知({stat['bzzt']})")
                print(f"  bzzt={stat['bzzt']} ({bzzt_label}): {stat['count']} 条")

            print("=" * 100)

        conn.close()

    except Exception as e:
        print(f"查询失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_latest_records()
