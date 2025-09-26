#include "misora2_cracks/size.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// get_line_widthの定義（staticメンバ関数）
double CracksSize::get_line_width(const cv::Mat& gray, const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Point2f dir = p2 - p1;
    float len = cv::norm(dir);
    if (len == 0) return 0.0f;
    dir *= 1.0f / len;

    cv::Point2f normal(-dir.y, dir.x);
    cv::Point2f center = (p1 + p2) * 0.5f;

    auto sample = [&](const cv::Point2f& q)->int {
        int x = std::clamp(cvRound(q.x), 0, gray.cols - 1);
        int y = std::clamp(cvRound(q.y), 0, gray.rows - 1);
        return (int)gray.at<uchar>(y, x);
    };

    int Ic = sample(center);

    int bgCnt = 0; double Ibg_sum = 0.0;
    for (int d = 15; d <= 25; ++d) {
        cv::Point2f ppos = center + normal * (float)d;
        cv::Point2f pneg = center - normal * (float)d;
        if (ppos.x >= 0 && ppos.x < gray.cols && ppos.y >= 0 && ppos.y < gray.rows) {
            Ibg_sum += sample(ppos); bgCnt++;
        }
        if (pneg.x >= 0 && pneg.x < gray.cols && pneg.y >= 0 && pneg.y < gray.rows) {
            Ibg_sum += sample(pneg); bgCnt++;
        }
    }
    if (bgCnt == 0) return 0.0;
    double Ibg = Ibg_sum / bgCnt;

    double T = 0.5 * (Ic + Ibg);
    int margin = 10;  // 厳しさを調整（5〜10推奨）

    int max_check = 30;
    int width_pos = 0, width_neg = 0;

    for (int d = 1; d <= max_check; ++d) {
        cv::Point2f p = center + normal * (float)d;
        if (p.x < 0 || p.x >= gray.cols || p.y < 0 || p.y >= gray.rows) break;
        int I = sample(p);
        if (I <= T - margin) width_pos++;
        else break;
    }
    for (int d = 1; d <= max_check; ++d) {
        cv::Point2f p = center - normal * (float)d;
        if (p.x < 0 || p.x >= gray.cols || p.y < 0 || p.y >= gray.rows) break;
        int I = sample(p);
        if (I <= T - margin) width_neg++;
        else break;
    }

    return static_cast<double>(width_pos + width_neg);
}
// detect_LSDの定義
std::vector<CracksSize::LineInfo> CracksSize::detect_LSD(const cv::Mat& original, int blur_size, int nfa_thresh) {
    cv::Mat gray;
    if (original.channels() == 3) cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    else gray = original;

    cv::Mat proc = gray.clone();
    int k = std::max(1, blur_size) | 1;
    cv::GaussianBlur(proc, proc, cv::Size(k, k), 1.0);

    std::vector<double> dat(proc.rows * proc.cols);
    for (int y = 0; y < proc.rows; ++y)
        for (int x = 0; x < proc.cols; ++x)
            dat[y * proc.cols + x] = (double)proc.at<uchar>(y, x);

    int n_lines = 0;
    double* lines_data = lsd(&n_lines, dat.data(), proc.cols, proc.rows);

    double scale_x = 20.0 / proc.cols;
    double scale_y = 20.0 / proc.rows;

    std::vector<LineInfo> all_lines;
    for (int i = 0; i < n_lines; ++i) {
        if (lines_data[i * 7 + 6] > nfa_thresh) {
            cv::Point2f p1(lines_data[i * 7 + 0], lines_data[i * 7 + 1]);
            cv::Point2f p2(lines_data[i * 7 + 2], lines_data[i * 7 + 3]);

            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            float angle_deg = std::fmod(std::abs(std::atan2(dy, dx)) * 180.0f / CV_PI, 180.0f);

            double dx_cm = dx * scale_x;
            double dy_cm = dy * scale_y;
            double length_mm = std::sqrt(dx_cm * dx_cm + dy_cm * dy_cm) * 10.0;

            bool is_angle_ok =
                std::abs(angle_deg - 0) < 5.0 ||
                std::abs(angle_deg - 45) < 5.0 ||
                std::abs(angle_deg - 90) < 5.0;

            if (is_angle_ok && length_mm >= 20.0 && length_mm <= 200.0) {
                double width_px = get_line_width(gray, p1, p2);
                double width_mm = width_px * (20.0 / proc.cols) * 10.0;
                if (width_mm >= 0.1 && width_mm < 5)
                    all_lines.push_back({p1, p2, length_mm, width_mm});
            }
        }
    }

    std::sort(all_lines.begin(), all_lines.end(),
              [](const LineInfo& a, const LineInfo& b) { return a.length > b.length; });

    return all_lines;
}

// find_bestの定義
CracksSize::Result CracksSize::find_best(const cv::Mat& original) {
    std::vector<CracksSize::Result> results;

    for (int blur = 1; blur <= 5; blur += 2) {     
        for (int nfa = 10; nfa <= 100; nfa += 5) {
            auto lines = detect_LSD(original, blur, nfa);
            double total_len = 0.0;
            double total_wid = 0.0;
            for (auto& l : lines) {
                total_len += l.length;
                total_wid += l.width;
            }
            results.push_back({blur, nfa, (int)lines.size(), total_len, total_wid});
        }
    }

    CracksSize::Result best = {};
    bool found = false;
    for (auto& r : results) {
        if (r.num_lines == 0) continue;
        if (!found || r.num_lines < best.num_lines ||
            (r.num_lines == best.num_lines && r.total_length > best.total_length)) {
            best = r;
            found = true;
        }
    }

    return found ? best : CracksSize::Result{ 0, 0, 0, 0.0, 0.0 };
}

// draw_linesの定義（非staticメンバ関数）
void CracksSize::draw_lines(const cv::Mat& original, const std::vector<CracksSize::LineInfo>& lines,
    int blur_size) {
        cv::Mat gray;
    if (original.channels() == 3) cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    else gray = original;

    cv::Mat display = gray.clone();
    int k = std::max(1, blur_size) | 1;
    cv::GaussianBlur(display, display, cv::Size(k, k), 1.0);

    cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);

    for (auto& l : lines) {
        cv::line(display, l.p1, l.p2, cv::Scalar(0, 0, 255), 2);
        char text[64];
        snprintf(text, sizeof(text), "L%.1fmm W%.1fmm", l.length, l.width);
        cv::putText(display, text, l.p1 + cv::Point2f(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("result_image", display);
    cv::waitKey(0);
}

// homographyの定義（staticメンバ関数）
cv::Mat CracksSize::homography(const cv::Mat& input_gray, float shrink_ratio) {
    if (input_gray.empty() || input_gray.channels() != 1) {
        std::cerr << "Input must be a grayscale image.\n";
        return cv::Mat();
    }

    cv::Mat binary;
    cv::threshold(input_gray, binary, 100, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        std::cerr << "No contours found.\n";
        return cv::Mat();
    }

    size_t max_idx = 0;
    double max_area = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }

    const std::vector<cv::Point>& contour = contours[max_idx];

    cv::Point tl, tr, br, bl;
    double min_sum = 1e9, max_sum = -1e9, min_diff = 1e9, max_diff = -1e9;
    for (const auto& pt : contour) {
        int sum = pt.x + pt.y;
        int diff = pt.x - pt.y;
        if (sum < min_sum) { min_sum = sum; tl = pt; }
        if (sum > max_sum) { max_sum = sum; br = pt; }
        if (diff < min_diff) { min_diff = diff; bl = pt; }
        if (diff > max_diff) { max_diff = diff; tr = pt; }
    }

    std::vector<cv::Point2f> quad_pts = { tl, bl, br, tr };

    cv::Point2f center(0, 0);
    for (const auto& pt : quad_pts) center += pt;
    center *= 1.0f / quad_pts.size();

    std::vector<cv::Point2f> shrunk_quad;
    for (const auto& pt : quad_pts) {
        cv::Point2f dir = center - pt;
        shrunk_quad.push_back(pt + dir * shrink_ratio);
    }

    float widthA = cv::norm(quad_pts[0] - quad_pts[11]);
    float widthB = cv::norm(quad_pts[12] - quad_pts[13]);
    float width = (widthA + widthB) / 2.0f;

    float heightA = cv::norm(quad_pts[0] - quad_pts[12]);
    float heightB = cv::norm(quad_pts[11] - quad_pts[13]);
    float height = (heightA + heightB) / 2.0f;

    int warped_width = std::round(width);
    int warped_height = std::round(height);

    std::vector<cv::Point2f> dst_pts = {
        {0, 0},
        {0, float(warped_height - 1)},
        {float(warped_width - 1), float(warped_height - 1)},
        {float(warped_width - 1), 0}
    };

    cv::Mat M = cv::getPerspectiveTransform(shrunk_quad, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(input_gray, warped, M, cv::Size(warped_width, warped_height));

    return warped;
}

// run_detectionの定義
std::tuple<CracksSize::Result, std::vector<CracksSize::LineInfo>, cv::Mat> CracksSize::run_detection(const cv::Mat& original) {
    // cv::Mat corrected = homography(original);
    cv::Mat corrected = original.clone();
    if (corrected.empty()) {
        std::cerr << "Projection transform failed.\n";
        cv::Mat zero = cv::Mat::zeros(480, 640, CV_8UC1);
        return std::make_tuple(CracksSize::Result{}, std::vector<CracksSize::LineInfo>{}, zero);
    }
    // cv::imshow("corrected",corrected);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    CracksSize::Result best = find_best(corrected);
    if (best.num_lines == 0) {
        std::cerr << "No valid result.\n";
        return std::make_tuple(CracksSize::Result{}, std::vector<CracksSize::LineInfo>{}, corrected);
    }

    std::vector<CracksSize::LineInfo> lines = detect_LSD(corrected, best.blur, best.nfa);
    return std::make_tuple(best, lines, corrected);
}
