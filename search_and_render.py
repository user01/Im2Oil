import json, math, os, tqdm
from Line_Tools import *
from simulate_RGB import *
from drawpatch import *


def Search_Stroke(
    anchors,
    density,
    img_gray,
    input_hsv,
    gradient,
    RATIO,
    angle,
    angle_hatch,
    min_width,
    min_length,
    max_width,
    max_length,
    threshold_hsv,
):
    (h0, w0) = img_gray.shape
    patch_sequence = []
    for i in tqdm.tqdm(range(anchors.shape[0])):
        patch_temp = {}
        # get coordinate
        (x, y) = anchors[i]
        y = h0 - y  # y is Upside-down ！！！
        y = np.clip(y, 0, h0 - 1)
        x = np.clip(x, 0, w0 - 1)
        x = int(round(x))  # for savety
        y = int(round(y))  # for savety

        patch_temp["coordinate"] = [y, x]

        patch_temp["angle_hatch"] = angle_hatch[y, x]
        patch_temp["path_hatch"], patch_temp["w1"], patch_temp["w2"] = FindPath_V7(
            img_gray,
            input_hsv,
            angle_hatch,
            y,
            x,
            step_length=1,
            min_length=min_width,
            max_length=max_width,
            threshold_hsv=threshold_hsv,
        )

        patch_temp["angle_ETF"] = angle[y, x]
        patch_temp["path_ETF"], patch_temp["l1"], patch_temp["l2"] = FindPath_V7(
            img_gray,
            input_hsv,
            angle,
            y,
            x,
            step_length=1,
            min_length=min_length,
            max_length=max_length,
            threshold_hsv=threshold_hsv,
        )

        patch_temp["grayscale"] = img_gray[y, x]
        patch_temp["hsv"] = input_hsv[y, x]

        patch_temp["gradient"] = gradient[y, x]
        patch_temp["density"] = density[y, x]

        max_local_width = np.sqrt(1.0 / patch_temp["density"])
        patch_temp["w1"] = min(patch_temp["w1"], max_local_width)
        patch_temp["w1"] = max(patch_temp["w1"], min_width)
        patch_temp["w2"] = min(patch_temp["w2"], max_local_width)
        patch_temp["w2"] = max(patch_temp["w2"], min_width)
        patch_temp["l1"] = min(patch_temp["l1"], RATIO * max_local_width)
        patch_temp["l1"] = max(patch_temp["l1"], min_length)
        patch_temp["l2"] = min(patch_temp["l2"], RATIO * max_local_width)
        patch_temp["l2"] = max(patch_temp["l2"], min_length)

        patch_temp["importance"] = (patch_temp["w1"] + patch_temp["w2"] + 1) * (
            patch_temp["l1"] + patch_temp["l2"] + 1
        ) + np.random.randn() / 100

        patch_sequence.append(patch_temp)

    return patch_sequence


def Render_Stroke(
    brush,
    patch_sequence,
    img_gray,
    output_path,
    max_length,
    SSAA,
    BORDERCOPY,
    FREQ,
    save=True,
    save_strokes=False,
    stroke_index_offset=0,
    canvas=None,
    mask=None,
):
    (h0, w0) = img_gray.shape
    if canvas is not None:
        Canvas = canvas
    else:
        Canvas = Gassian_HSV(
            (h0 * SSAA + 2 * max_length * SSAA, w0 * SSAA + 2 * max_length * SSAA, 3)
        )  # padding max_length
    if mask is not None:
        Mask = mask
    else:
        Mask = np.zeros(
            (h0 * SSAA + 2 * max_length * SSAA, w0 * SSAA + 2 * max_length * SSAA)
        )
    stroke_metadata = []
    for step in tqdm.tqdm(range(len(patch_sequence))):
        patch_temp = patch_sequence[step]
        w1 = patch_temp["w1"]
        w2 = patch_temp["w2"]
        l1 = patch_temp["l1"]
        l2 = patch_temp["l2"]
        length = int(round((l1 + 1 + l2) * SSAA))
        width = int(round((w1 + 1 + w2) * SSAA))

        angle_ETF = patch_temp["angle_ETF"]
        angle_Hatch = angle_Hatch = patch_temp["angle_hatch"]

        # get patch central coordinate
        central = patch_temp["coordinate"]
        central[1] -= (l1 - l2) / 2 * math.cos(angle_ETF / 180 * math.pi)  # x
        central[0] += (l1 - l2) / 2 * math.sin(angle_ETF / 180 * math.pi)  # y
        central[1] -= (w1 - w2) / 2 * math.cos(angle_Hatch / 180 * math.pi)  # x
        central[0] += (w1 - w2) / 2 * math.sin(angle_Hatch / 180 * math.pi)  # y

        Y, X = int(round(central[0] * SSAA)), int(round(central[1] * SSAA))
        Y += max_length * SSAA  # Canvas edge is padded
        X += max_length * SSAA  # Canvas edge is padded

        hsv = patch_temp["hsv"]

        stroke, mask = GetBrush_HSV(width=width, length=length, mean=hsv, brush=brush)
        rotate_stroke, _ = rotate_hsv(
            stroke, angle_ETF, pad_color=(int(hsv[0]), 0, 255)
        )
        rotate_stroke = np.clip(rotate_stroke, 0, 255)
        (rotate_h, rotate_w, _) = rotate_stroke.shape
        rotate_hh, rotate_ww = int(rotate_h / 2), int(rotate_w / 2)

        mask, _ = rotate(mask, angle_ETF, pad_color=0)
        mask[mask < 1] = 0
        # cv2.imshow('Canvas', np.uint8(rotate_stroke))
        # cv2.waitKey(0)

        Mask[
            Y - rotate_hh : Y - rotate_hh + rotate_h,
            X - rotate_ww : X - rotate_ww + rotate_w,
        ] += mask
        temp = Canvas[
            Y - rotate_hh : Y - rotate_hh + rotate_h,
            X - rotate_ww : X - rotate_ww + rotate_w,
        ]

        rotate_alpha = mask.astype("float32")
        rotate_alpha = rotate_alpha[:, :, np.newaxis]
        rotate_stroke = rotate_stroke.astype("float32")

        if save_strokes:
            global_idx = stroke_index_offset + step
            stroke_bgr = cv2.cvtColor(
                np.uint8(np.clip(rotate_stroke, 0, 255)), cv2.COLOR_HSV2BGR
            )
            alpha_channel = np.uint8(mask * 255)
            stroke_bgra = cv2.merge([
                stroke_bgr[:, :, 0],
                stroke_bgr[:, :, 1],
                stroke_bgr[:, :, 2],
                alpha_channel,
            ])
            cv2.imwrite(
                output_path + "/stroke/{0:05d}.png".format(global_idx),
                stroke_bgra,
            )
            # Placement relative to the final cropped output
            crop_offset = max_length * SSAA + BORDERCOPY * SSAA
            stroke_metadata.append(
                {
                    "index": global_idx,
                    "x": int(X - rotate_ww - crop_offset),
                    "y": int(Y - rotate_hh - crop_offset),
                    "width": int(rotate_w),
                    "height": int(rotate_h),
                    "angle": float(angle_ETF),
                    "hsv": [int(hsv[0]), int(hsv[1]), int(hsv[2])],
                    "importance": float(patch_temp.get("importance", 0)),
                    "grayscale": int(patch_temp.get("grayscale", 0)),
                }
            )

        Canvas[
            Y - rotate_hh : Y - rotate_hh + rotate_h,
            X - rotate_ww : X - rotate_ww + rotate_w,
        ] = np.uint8((1 - rotate_alpha) * temp + rotate_alpha * rotate_stroke)

        if save and (step + 1) % FREQ == 0:
            result = Canvas[
                max_length * SSAA : -max_length * SSAA,
                max_length * SSAA : -max_length * SSAA,
            ]
            cv2.imwrite(
                output_path + "/process/{0:05d}.png".format(step + 1),
                cv2.cvtColor(
                    result[
                        BORDERCOPY * SSAA : -BORDERCOPY * SSAA,
                        BORDERCOPY * SSAA : -BORDERCOPY * SSAA,
                    ],
                    cv2.COLOR_HSV2BGR,
                ),
            )
            # cv2.imshow('step', Canvas)
            # cv2.waitKey(0)
    if save and step == len(patch_sequence) - 1 and step + 1 % FREQ != 0:  # 存最后一张
        result = Canvas[
            max_length * SSAA : -max_length * SSAA,
            max_length * SSAA : -max_length * SSAA,
        ]
        cv2.imwrite(
            output_path + "/process/{0:05d}.png".format(step + 1),
            cv2.cvtColor(
                result[
                    BORDERCOPY * SSAA : -BORDERCOPY * SSAA,
                    BORDERCOPY * SSAA : -BORDERCOPY * SSAA,
                ],
                cv2.COLOR_HSV2BGR,
            ),
        )

    Mask[Mask > 0] = 1
    # cv2.imshow('mask', np.uint8(Mask*255))
    # cv2.waitKey(0)
    return Canvas, Mask, stroke_metadata  # hsv
