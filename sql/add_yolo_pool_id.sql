-- =====================================================
-- 数据库迁移脚本：为摄像头表添加YOLO池ID字段
-- 用途：支持灵活配置哪些摄像头共享同一个YOLO实例
-- =====================================================

-- 1. 为 app_roomip 表添加 yolo_pool_id 字段
ALTER TABLE app_roomip
ADD COLUMN yolo_pool_id INT DEFAULT NULL COMMENT 'YOLO实例池ID: NULL或-1=使用全局默认池, 0=独立实例, 1/2/3...=使用指定池ID（相同ID共享实例）';

-- 2. 添加索引（可选，提升查询性能）
CREATE INDEX idx_yolo_pool_id ON app_roomip(yolo_pool_id);

-- =====================================================
-- 示例配置数据
-- =====================================================

-- 示例1：摄像头1和2共享池1
-- UPDATE app_roomip SET yolo_pool_id = 1 WHERE id IN (1, 2);

-- 示例2：摄像头3和4共享池2
-- UPDATE app_roomip SET yolo_pool_id = 2 WHERE id IN (3, 4);

-- 示例3：摄像头5使用独立实例
-- UPDATE app_roomip SET yolo_pool_id = 0 WHERE id = 5;

-- 示例4：摄像头6使用全局默认池（config.json配置）
-- UPDATE app_roomip SET yolo_pool_id = NULL WHERE id = 6;

-- =====================================================
-- 查询配置
-- =====================================================

-- 查看所有摄像头的YOLO池配置
-- SELECT id, fjmc, yolo_pool_id FROM app_roomip ORDER BY yolo_pool_id, id;

-- 统计每个池有多少个摄像头
-- SELECT
--     CASE
--         WHEN yolo_pool_id IS NULL THEN '全局默认池'
--         WHEN yolo_pool_id = 0 THEN '独立实例'
--         ELSE CONCAT('池', yolo_pool_id)
--     END AS pool_name,
--     COUNT(*) AS camera_count
-- FROM app_roomip
-- GROUP BY yolo_pool_id
-- ORDER BY yolo_pool_id;
