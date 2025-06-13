/*
 * Copyright 2025 Naturalis Biodiversity Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
function addDots(data, x, y, color_by, transform, width, height) {
    const [show_data, k] = filterData(data, 10000, transform, width, height);
    console.log("addDots", show_data.length);
    // Add dots
    dots = g
        .append("g")
        .attr("id", "dots")
        .selectAll("dot")
        .data(show_data)
        .enter()
        .append("circle")
        .attr("cx", function (d) {
            return x(d.e_0);
        })
        .attr("cy", function (d) {
            return y(d.e_1);
        })
        .attr("r", 3.0 / k)
        .attr("id", function (d) {
            return "dot-" + d.uid;
        })
        .style("fill", function (d) {
            return color_by ? d[color_by] : d.color_class;
        })
        .style("stroke", function (d) {
            return d.color_prob;
        })
        .style("stroke-width", function (d) {
            return (d.labeled == 0 ? 2 * (1 - d.score_predicted) : 2) / k;
        });
}

function addTimeline(data, transform, width, height) {
    const [show_data, k] = filterData(data, 1000, transform, width, height); // TODO: same data as dots
    // Add dots
    let bars = g
        .append("g")
        .attr("id", "time_bars")
        .selectAll("bar")
        .data(show_data)
        .enter()
        .append("line")
        .attr("x1", function (d) {
            return time_x(d.time_s);
        })
        .attr("x2", function (d) {
            return time_x(d.time_s);
        })
        .attr("y1", function (d) {
            return height;
        })
        .attr("y2", function (d) {
            return height - d.score_predicted * 50;
        })
        .attr("id", function (d) {
            return "time_bar-" + d.time_s;
        })
        .attr("stroke", function (d) {
            return d.color_class;
        })
        .style("stroke-width", function (d) {
            return 1;
        });
    return bars;
}

function filterData(data, max_n, transform, width, height) {
    const now = new Date().getTime() / 1000;
    let k = 1;
    let Tx = 1;
    let Ty = 1;
    console.log("transform", transform);
    if (transform !== undefined && transform) {
        k = transform.k;
        Tx = transform.x;
        Ty = transform.y;
    }
    let show_data_start = [...data];
    show_data_start.sort((a, b) => a.display_order - b.display_order);
    const label_predicted = urlParams.get("label_predicted");
    if (label_predicted) {
        show_data_start = show_data_start.filter((a) =>
            // a["label_predicted"].toLowerCase().includes(label_predicted)
            urlParams.get("not_label_predicted") == "on"
                ? !a["label_predicted"].toLowerCase().includes(label_predicted)
                : a["label_predicted"].toLowerCase().includes(label_predicted)
        );
    }
    const label_true = urlParams.get("label_true");
    if (label_true) {
        show_data_start = show_data_start.filter((a) =>
            urlParams.get("not_label_true") == "on"
                ? !a["label_true"].toLowerCase().includes(label_true)
                : a["label_true"].toLowerCase().includes(label_true)
        );
    }
    const label_undetermined = urlParams.get("label_undetermined");
    if (label_undetermined) {
        show_data_start = show_data_start.filter((a) =>
            a["label_undetermined"].toLowerCase().includes(label_undetermined)
        );
    }
    let show_data = [];
    // alert(width, height);
    for (const d of show_data_start) {
        const tX = x(d.e_0) * k + Tx;
        const tY = y(d.e_1) * k + Ty;
        // console.log(tX, tY);
        if (tX > 0 && tX < width && tY > 0 && tY < height) {
            show_data.push(d);
        }
        if (show_data.length >= max_n) {
            break;
        }
    }
    show_data = show_data.reverse();
    console.log("show_data.length", show_data.length);
    // console.log("filterData took", new Date().getTime() / 1000 - now)
    return [show_data, k];
}


function addImages(data, x, y, transform, width, height) {
    const [show_data, k] = filterData(data, 50, transform, width, height);
    g.append("g")
        .attr("id", "images")
        .selectAll("images")
        .data(show_data)
        .enter()
        .append("image")
        .attr("x", function (d) {
            return x(d.e_0);
        })
        .attr("y", function (d) {
            return y(d.e_1);
        })
        .attr("width", function (d) {
            return 128 / k;
        })
        .attr("height", function (d) {
            return 128 / k;
        })
        .attr("href", function (d) {
            return "/images/thumbnail/" + d.uid;
        })
        .attr("id", function (d) {
            return "map-" + d.uid;
        })
        .attr("data-label", function (d) {
            return d.label_true;
        });

    d3.selectAll("image").on("click", selectImage);
}

function clearImages() {
    d3.select("#images").remove();
}

function round(num, places) {
    const factor = Math.pow(10, places);
    return Math.round((num + Number.EPSILON) * factor) / factor;
}

function invertMap(childToParentMap) {
    const parentToChildMap = new Map();

    for (const [child, parent] of childToParentMap) {
        if (!parentToChildMap.has(parent)) {
            parentToChildMap.set(parent, []);
        }
        parentToChildMap.get(parent).push(child);
    }

    return parentToChildMap;
}

function renderPerformance(data) {
    console.log("performance");
    const test_performance = data["test_performance"];
    const diff =
        test_performance[test_performance.length - 1][2] -
        test_performance[test_performance.length - 2][2];
    $("#test_performance").html(
        round(test_performance[test_performance.length - 1][2] * 100.0, 1) +
        " (" +
        round(diff * 100.0, 1) +
        ") %"
    );
    $("#percentage_near_labeled").html(
        round(data["percentage_near_labeled"] * 100.0, 1)
    );
}

function excludeDescendants(left, right, parentToChildren, exclusivity) {
    if (parentToChildren.has(left)) {
        for (const child of parentToChildren.get(left)) {
            if (!exclusivity.get(child)) {
                exclusivity.set(child, []);
            }
            for (const el of right) {
                exclusivity.get(child).push(el);
                if (!exclusivity.get(el)) {
                    exclusivity.set(el, []);
                }
                exclusivity.get(el).push(child);
            }
        }
    }
}

controlHtml = `<div id="map_control">
      <table>
      <tr>
          <td>Status <img src="/static/loading_circle.gif" width="58px" id="loading_circle" style="display: none;"/></td>
          <td>
            <span id="status"></span>
            <div id="status" style="width:100%; border: 1px solid white; display: flex; position: relative; height:1em">
              <div id="progress" style="background-color:darkgreen; width:100%; position: absolute; height: 100%; bottom:0; top:0"></div>
            </div>
          </td>
        </tr>
        <tr>
          <td><img src="/static/annflux.svg" width="32px"/></td>
          <td><span id="speed_active_time"></span> active, <span id="speed_annotation"></span> <img src="/static/label.svg" width="24px"/>/h, likely certain <span id="speed_certain"></span><img src="/static/label.svg" width="24px"/>/h</td>
        
        <tr>
          <td>Quick training</td>
          <td>
            <a href="javascript:void(0)" onclick="forceLinearTrain()"
              >Train now</a
            >
          </td>
        </tr>
        <tr>
          <td>
            <a href="/detailed_performance">Performance</a>
          </td>
          <td>
            <span id="test_performance"></span>
          </td>
        </tr>
        <tr>
          <td>Space covered</td>
          <td>
            <span id="percentage_near_labeled"></span> %
          </td>
        </tr>
        <tr>
          <td>UI log</td>
          <td>
            <span id="ui_log_last"></span>
          </td>
        </tr>

      </table>
      <a href="#" onclick="persistentToggle('view_config')">View <img src="/static/settings.svg" width="16px"/></a> <a href="/annflux">Reset</a>
      <table id="view_config">

        <tr>
          <td width="300px">Ranking</td>
          <td>
            <select id="as_ranking_column" onchange="changeOption(this)">
              <option value="score_predicted">
                Prediction uncertainty
              </option>
              <option value="high_label_entropy">High label entropy</option>
              <option value="score_true">True probability</option>
              <option value="fre">FRE</option>
              <option value="certain_incorrect">Certain incorrect</option>
              <option value="most_needed" selected="selected">Most needed</option>
              <option value="incorrect_score">Incorrect score</option></select
            ><input
              type="checkbox"
              id="invert_ranking"
              name="invert_ranking"
              onchange="changeOption(this)"
            />
            <label for="invert_ranking">INVERT</label>
          </td>
        </tr>
        <tr>
          <td>Label predicted</td>
          <td>
            <select id="label_predicted" onchange="changeOption(this)"></select>
            <input
              type="checkbox"
              id="not_label_predicted"
              name="not_label_true"
              onchange="changeOption(this)"
            />
            <label for="not_label_predicted">NOT</label>
          </td>
        </tr>
        <tr>
          <td>Label true</td>
          <td>
            <select id="label_true" onchange="changeOption(this)"></select>
            <input
              type="checkbox"
              id="not_label_true"
              name="not_label_true"
              onchange="changeOption(this)"
            />
            <label for="not_label_true">NOT</label>
          </td>
        </tr>
        <tr>
          <td>Label undetermined</td>
          <td>
            <select
              id="label_undetermined"
              onchange="changeOption(this)"
            ></select>
          </td>
        </tr>
        <tr>
          <td>Show</td>
          <td>
            <select id="show_labeled" onchange="changeOption(this)">
              <option value="unlabeled" selected="selected">unlabeled</option>
              <option value="labeled">labeled</option>
              <option value="both">both</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>Color</td>
          <td>
            <select id="color_map" onchange="changeOption(this)">
              <option value="color_class" selected="selected">
                Predicted class
              </option>
              <option value="color_prob">Probability</option>
              <option value="color_fre">FRE</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>Num in gallery</td>
          <td>
            <select id="num_in_gallery" onchange="changeOption(this)">
              <option value="9" selected="selected">10</option>
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="100">100</option>
              <option value="200">200</option>
              <option value="500">500</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>Flux mode</td>
          <td>
            <select id="flux_mode" onchange="changeOption(this)">
              <option value="active_learning" selected="selected">Active learning</option>
              <option value="active_learning_shard">Shard active learning</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>
            <label for="ignore_double_checked">Ignore double checked</label>
          </td>
          <td>
            <input
              type="checkbox"
              id="ignore_double_checked"
              name="ignore_double_checked"
              onchange="changeOption(this)"
            />
          </td>
        </tr>
      </table>
    </div>`


const mapHtml = `<div id="my_dataviz" tabindex="0"></div> 
    <div id="help" style="position: fixed; max-width: 60%; background-color: #333333cc; display: none; z-index:100">
        <div id="annflux_title">
      <span style="color:rgb(134, 218, 25)" class="title_caps">IN</span>teractive <span style="color:rgb(230, 226, 16)" class="title_caps">D</span>ata <span style="color:rgb(163, 44, 147)" class="title_caps">E</span>xploration and <span style="color:rgb(43, 119, 170)" class="title_caps">E</span>nrichment <span style="color:rgb(211, 46, 17)" class="title_caps">D</span>evice
    </div>
      <table>
        <tr>
          <td width="16%"><b>Map</b></td>
          <td width="16%"></td>
          <td width="16%"><b>Gallery</b></td>
          <td width="16%"></td>
          <td width="16%"><b>General</b></td>
          <td width="16%"></td>
        </tr>
        <tr>
          <td></td>
          <td></td>
          <td></td>
          <td></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <td>Mouse scroll</td>
          <td><i>Zoom</i></td>
          <td>Click</td>
          <td><i>Select single image</i></td>
          <td>?</td>
          <td><i>Toggle this help screen</i></td>
        </tr>
        <tr>
          <td>Click & drag</td>
          <td><i>Pan</i></td>
          <td>Shift-click</td>
          <td><i>Select multiple images</i></td>
          <td></td>
          <td><i></i></td>
        </tr>
        <tr>
          <td>Ctrl+Click & drag</td>
          <td><i>Select images</i></td>
          <td>Label shortcuts (hover over labels on top)</td>
          <td><i>Label by keyboard</i></td>
          <td></td>
          <td><i></i></td>
        </tr>
        <tr>
          <td>i</td>
          <td><i>disable thumbnails</i></td>
          <td>Ctrl+click <span class="inline-help" title="the labels suggested by the tool">possible</span> label</td>
          <td><i>Make label <span class="inline-help" title="whether the label applies is unsure">undetermined</span></i></td>
          <td>Version</td>
          <td><span id="package_version"></span></td>
        </tr>
      </table></div>`