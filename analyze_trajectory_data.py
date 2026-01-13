"""
分析轨迹数据 - 统计今天的轨迹和截图情况
"""
import pymysql
from datetime import datetime
import config_loader as config

def analyze_trajectory_data():
    """分析今天的轨迹数据"""
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

        today = datetime.now().strftime('%Y-%m-%d')

        print("=" * 100)
        print(f"轨迹数据分析 - {today}")
        print("=" * 100)

        with conn.cursor() as cursor:
            # 1. 统计今天的轨迹总数
            sql_total = """
                SELECT COUNT(*) as total
                FROM app_track
                WHERE DATE(pssj) = %s
            """
            cursor.execute(sql_total, (today,))
            total = cursor.fetchone()['total']
            print(f"\n[1] 今天的轨迹总数: {total} 条")

            # 2. 按摄像头统计
            sql_by_camera = """
                SELECT
                    qyid,
                    qymc,
                    sxtmx,
                    COUNT(*) as count,
                    SUM(jscs) as total_detections,
                    MIN(pssj) as first_time,
                    MAX(pssj) as last_time
                FROM app_track
                WHERE DATE(pssj) = %s
                GROUP BY qyid, qymc, sxtmx
                ORDER BY count DESC
            """
            cursor.execute(sql_by_camera, (today,))
            cameras = cursor.fetchall()

            print(f"\n[2] 按摄像头统计 (共 {len(cameras)} 个摄像头):")
            print("-" * 100)
            print(f"{'摄像头ID':<10} {'区域名称':<15} {'摄像头名称':<15} {'轨迹数':<10} {'总检测次数':<12} {'首次时间':<20} {'最后时间':<20}")
            print("-" * 100)

            for cam in cameras:
                print(f"{cam['qyid']:<10} {cam['qymc']:<15} {cam['sxtmx']:<15} {cam['count']:<10} {cam['total_detections']:<12} {str(cam['first_time']):<20} {str(cam['last_time']):<20}")

            # 3. 统计截图总数（app_track_screenshot表）
            try:
                sql_screenshots = """
                    SELECT COUNT(*) as total
                    FROM app_track_screenshot s
                    JOIN app_track h ON s.track_id = h.id
                    WHERE DATE(h.pssj) = %s
                """
                cursor.execute(sql_screenshots, (today,))
                screenshot_total = cursor.fetchone()['total']
                print(f"\n[3] 今天的截图总数 (app_track_screenshot): {screenshot_total} 张")
            except Exception as e:
                print(f"\n[3] 查询截图表失败 (可能表不存在): {e}")

            # 4. 统计每个轨迹的检测次数分布
            sql_jscs_dist = """
                SELECT
                    jscs,
                    COUNT(*) as count
                FROM app_track
                WHERE DATE(pssj) = %s
                GROUP BY jscs
                ORDER BY jscs DESC
            """
            cursor.execute(sql_jscs_dist, (today,))
            jscs_dist = cursor.fetchall()

            print(f"\n[4] 检测次数分布 (合并情况):")
            print("-" * 60)
            print(f"{'检测次数(jscs)':<20} {'轨迹数量':<20} {'说明':<20}")
            print("-" * 60)

            for item in jscs_dist:
                jscs = item['jscs']
                count = item['count']
                desc = "未合并(新轨迹)" if jscs == 1 else f"合并了{jscs-1}次检测"
                print(f"{jscs:<20} {count:<20} {desc:<20}")

            # 5. 找出检测次数最多的轨迹（可能是长时间活动）
            sql_top_jscs = """
                SELECT
                    id,
                    qymc,
                    sxtmx,
                    pssj,
                    jssj,
                    jscs,
                    rysl,
                    TIMESTAMPDIFF(SECOND, pssj, jssj) as duration_seconds
                FROM app_track
                WHERE DATE(pssj) = %s
                ORDER BY jscs DESC
                LIMIT 10
            """
            cursor.execute(sql_top_jscs, (today,))
            top_jscs = cursor.fetchall()

            print(f"\n[5] 检测次数最多的10条轨迹:")
            print("-" * 100)
            print(f"{'轨迹ID':<10} {'区域':<15} {'摄像头':<15} {'开始时间':<20} {'检测次数':<10} {'持续时长(秒)':<15}")
            print("-" * 100)

            for item in top_jscs:
                print(f"{item['id']:<10} {item['qymc']:<15} {item['sxtmx']:<15} {str(item['pssj']):<20} {item['jscs']:<10} {item['duration_seconds']:<15}")

            # 6. 统计标注状态
            sql_bzzt = """
                SELECT
                    bzzt,
                    COUNT(*) as count
                FROM app_track
                WHERE DATE(pssj) = %s
                GROUP BY bzzt
            """
            cursor.execute(sql_bzzt, (today,))
            bzzt_stats = cursor.fetchall()

            print(f"\n[6] 标注状态统计:")
            print("-" * 40)
            for stat in bzzt_stats:
                bzzt_val = stat['bzzt'] if stat['bzzt'] is not None else 'NULL'
                bzzt_label = "待标注" if bzzt_val == '0' else ("已标注" if bzzt_val == '1' else f"未知({bzzt_val})")
                print(f"  bzzt={bzzt_val} ({bzzt_label}): {stat['count']} 条")

            # 7. 计算理论检测总数（如果每次都创建新轨迹应该有多少条）
            sql_total_detections = """
                SELECT SUM(jscs) as total_detections
                FROM app_track
                WHERE DATE(pssj) = %s
            """
            cursor.execute(sql_total_detections, (today,))
            total_detections = cursor.fetchone()['total_detections']

            print(f"\n[7] 检测汇总:")
            print(f"  实际轨迹数: {total} 条")
            print(f"  总检测次数: {total_detections} 次")
            print(f"  平均每条轨迹被检测: {total_detections/total:.1f} 次" if total > 0 else "  无数据")
            print(f"  合并比例: {(1 - total/total_detections)*100:.1f}%" if total_detections > 0 else "  无数据")
            print(f"\n  说明: Python检测到人{total_detections}次，但由于30秒合并窗口，只创建了{total}条轨迹")

            print("\n" + "=" * 100)
            print("分析完成")
            print("=" * 100)

        conn.close()

    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    analyze_trajectory_data()
