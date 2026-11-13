# Multi-blot Figure (approved 2026-11-12, mockup boards 10 and 11)

> **Superseded (2026-11-13):** the owner chose the Board B mockup instead of this picker sheet
> (https://claude.ai/artifact/3FWepE9YqtPm7qxDge4GTv). 1.1.0 implements Board B: image check boxes
> and protein chips under '그림에 넣을 블롯' in the Figure sidebar, lanes matched by lane name by
> default (or 순서대로), an on/off source tag under each protein name, and per-panel cards in the
> inspector. Data: `panels[]` entries with `doc` (absolute image path), `src_row`, optional
> `lane_map`, `gamma`, `height_k`, `auto_range`; figure settings `lane_match`, `show_src`, `image_gap_pt`.

Mockup canvas: https://claude.ai/artifact/KSQyW4Uao78mCT6dZ2BaCf
- Board 10: "Figure 창 · 여러 이미지의 블롯을 한 그림에".
- Board 11: "‘다른 이미지의 블롯…’ 고르기 창".

The user approved this together with:
- The 1.0.2 screen fixes.
- Band detection that follows the red protein lines.

## What the user sees

### Sidebar (window floor)

**'추가' list.** The first row is new: **다른 이미지의 블롯…**. It is in accent colour and carries a "새로" note. It opens the picker sheet.

**'그림 안의 것' list.** It is grouped by image:
- A small grey header per image, for example `I1 · 10.15_Ch.tif`.
- Under each header, its protein panels.
- Then a `글자·숫자` group with lane names / group bars and number rows.

Clicking a row selects that object in the preview, the same as clicking it in the preview.

### Preview (레인별 자르기 mode)

Panels from several open images stack in one figure, in the chosen order.
- Each panel keeps its own crop, contrast and pad settings from its source image.
- Where the next image starts there is a slightly larger gap. This is the inspector's '이미지 사이 간격', default 6 pt.
- A small grey source tag (I1, I2) right of each panel shows only in the preview. It is never exported.
- Lane names, group bars and kDa ticks come from the top (first) image.
- Number rows may use any image's quant.

### Inspector, 선택한 대상, panel from another image

**Card 1:**
- 이미지 (pop-up of the open images).
- 단백질 (pop-up of that image's rows).
- 라벨.

**Card '레인 맞추기':**
- 방식: 같은 순서 (default) / 직접 고르기.
- Chips "top-lane-name ← Ln" for each column. In 직접 고르기, clicking a chip lets you pick another lane or "빈 칸".
- 이미지 사이 간격 (pt).

**Card 3:**
- 위·아래 여백.
- 밝기 범위 자동.
- 순서 ↑ ↓.
- 빼기 (red).

### Picker sheet (opens from 추가 › 다른 이미지의 블롯…)

**Left column: 열린 이미지.**
- One row per open image: thumbnail, `I2 · name`, and "단백질 n개 · m개 고름".
- A "+ 다른 TIF 열기…" row opens one more image.

**Middle column: proteins of the selected image.**
- One checkbox per protein, with a note: Control / 정량 있음.
- A thumbnail of that image with the checked rows outlined as blue bands.
- '레인 맞추기' card:
  - 방식: 같은 순서 (L1 → 첫 레인).
  - 레인 수가 다를 때: 빈 칸으로 두기 (default) / 잘라내기.

**Right column: Figure에 들어갈 순서.**
- The chosen panels with their source tag. Drag to reorder.
- Note under it: "레인 이름·그룹 막대·kDa 눈금은 맨 위 이미지(I1) 것을 씁니다."

**Buttons:** 취소, and the primary "n개 패널 넣기".

## Data (suggestion)

`figure settings["panels"]` entries may carry:
- `"doc"`: a stable image id or path. Missing means the figure's own image.
- `"lane_map"`: a list, one entry per figure column, of a source lane index or None.

Rendering takes each panel's strip from its own doc's corrected image and analysis. Old projects with no `doc` keep working unchanged. Project save/load must store the references. If a referenced image is not open, show the panel greyed with "이미지를 열면 보입니다". Export (PNG/TIFF/PDF/SVG and the figure CSV) includes every panel.
