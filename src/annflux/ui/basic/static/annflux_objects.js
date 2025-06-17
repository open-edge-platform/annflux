let gReferences = new Map();

function Map2D($container) {
  this.$container = $container;
  this.markers = [];
  this.goldenContainer = null;
  this.zooming = false;
  this.goldenContainer = null;
  let transform = null;
  this.prevTransform = Object();
  this.color_by = null;
  let d3Group = null;
  let dots = null;
  this.images = null;
  let elementId = null;
  let svgId = null;
  let dotsId = null;
  let imagesId = null;
  let mode = null;
  this.filters = [];
  this.x = null;
  this.y = null;

  // Initialize the map
  this.init = function (goldenContainer, elementId_, mode_) {
    elementId = elementId_;

    this.goldenContainer = goldenContainer;
    if (mode_ == undefined) {
      mode = "embedding";
    } else {
      //TODO: check mode in known values
      mode = mode_;
    }
    console.log("init", elementId_, elementId, mode, mode_);

    //
    // set the dimensions and margins of the graph
    var margin = { top: 0, right: 0, bottom: 0, left: 0 };
    width = this.goldenContainer.width;
    height = this.goldenContainer.height;

    this.prevTransform.k = 1;
    this.prevTransform.x = 0;
    this.prevTransform.y = 0;
    this.color_by = urlParams.get("color_map"); //TODO: do not use global

    svgId = `${elementId.substr(1)}_svg`;
    dotsId = `${elementId.substr(1)}_dots`;
    imagesId = `${elementId.substr(1)}_images`;
    // console.log(svgId, dotsId, imagesId);
    //
    gReferences.set(elementId, this);

    svg = d3
      .select(elementId)
      .append("svg")
      .attr("id", svgId)
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom);
    d3Group = svg
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    gallery = d3.select("#gallery").append("div");
  };

  this.handleZoom = function (e) {
    transform = d3.zoomTransform(this);
    d3Group.attr("transform", transform);
    zoomControl(elementId, transform, lastZoomUpdate, numUpdatesActive);
  };

  this.setFilter = function (filters_) {
    this.filters = filters_;
  };

  this.setData = function (data) {
    console.log("setData", elementId, mode);
    if (data !== undefined) {
      this.data = data;
    } else {
      data = this.data;
    }
    this.x = d3.scaleLinear().domain([-20, 20]).range([0, width]);
    this.y = d3.scaleLinear().domain([-20, 20]).range([height, 0]);
    if (mode == "tiles") {
      this.x = d3.scaleLinear().domain([0, 4000]).range([0, width]); // TODO: from data
      this.y = d3.scaleLinear().domain([0, 4000]).range([0, width]); // TODO: different aspect ratios
    }

    this.render(data, x, y, mode);
    //TODO(refactor): addTimeline(data, transform, width, height);
    console.log("setData", elementId, this.images, dots);
    uiObject = this;

    zoom = d3
      .zoom()
      .filter(function filter(event) {
        return !d3.event.ctrlKey && !d3.event.button;
      })
      .on("zoom", this.handleZoom)
      .on("end", function () {
        zoomControl(elementId, transform, lastZoomUpdate, numUpdatesActive);
        console.log("zoom this", this);
        this.prevTransform = transform;
      });

    function initZoom() {
      d3.select(`#${svgId}`).call(zoom);
    }

    initZoom();

    let coords = [];
    const lineGenerator = d3.line();

    function drawPath(k) {
      d3.select("#lasso") // TODO(not global)
        .style("stroke", "white")
        .style("stroke-width", 2 / k)
        .style("fill", "#00000054")
        .attr("d", lineGenerator(coords));
    }

    function dragStart() {
      console.log("dragStart", this);
      coords = [];
      if (dots) {
        dots.attr("fill", "steelblue");
      } else if (images) {
        images.attr("fill", "steelblue");
      } else {
        alert("no target for drag");
      }
      d3.select("#lasso").remove();
      d3.select(`${elementId} g`).append("path").attr("id", "lasso");
    }

    function dragMove(event) {
      let mouseX = d3.event.x;
      let mouseY = d3.event.y;
      let k = 1;
      let Tx = 1;
      let Ty = 1;
      if (transform !== undefined && transform) {
        k = transform.k;
        Tx = transform.x;
        Ty = transform.y;
      }
      mouseX = (mouseX - Tx) / k;
      mouseY = (mouseY - Ty) / k;
      coords.push([mouseX, mouseY]);
      drawPath(k);
    }

    function dragEnd(event, elementId) {
      //console.log("dragEnd", event, elementId);
      let selectedElements = [];
      const uiObject = gReferences.get(elementId);
      const uiElements = dots ? dots : this.images;
      uiElements.each((d, i) => {
        let point = [uiObject.x(d.e_0), uiObject.y(d.e_1)];
        if (pointInPolygon(point, coords)) {
          d3.select("#dot-" + d.uid).attr("fill", "red");
          selectedElements.push(d.uid);
        }
      });
      //
      let show_data = [];
      for (const d of data) {
        if (selectedElements.includes(d.uid)) {
          show_data.push(d);
        }
      }
      // sort
      const rank_modifier = urlParams.get("invert_ranking") == "on" ? -1 : +1;
      let as_ranking_column = urlParams.get("as_ranking_column");
      show_data = show_data.sort(
        (a, b) => rank_modifier * (a[as_ranking_column] - b[as_ranking_column])
      );
      console.log("dragEnd", show_data.length, show_data);
      drawGallery(show_data);
    }
    if (mode == "embedding") {
      // auto suggestion
      let as_ranking_column = urlParams.get("as_ranking_column");
      console.log("as_ranking_column", as_ranking_column);
      if (as_ranking_column == null) {
        as_ranking_column = "most_needed";
      }
      let show_labeled = urlParams.get("show_labeled");
      if (show_labeled == null) {
        show_labeled = "unlabeled";
      }
      let show_data2 = [];
      let show_data_test = [];
      if (as_ranking_column == "score_true") {
        show_labeled = "labeled";
      }
      let [show_data_start, tmp_] = filterData(
        data,
        this.x,
        this.y,
        data.length,
        transform,
        width,
        height,
        [],
        "e_0",
        "e_1"
      );
      let num_in_gallery = urlParams.get("num_in_gallery");
      if (num_in_gallery == null) {
        num_in_gallery = 10;
      }
      // console.log("foekoezoe", show_data_start);
      if (
        as_ranking_column == "score_predicted" ||
        as_ranking_column == "score_true" ||
        as_ranking_column == "fre" ||
        as_ranking_column == "most_needed" ||
        as_ranking_column == "incorrect_score" ||
        as_ranking_column == "certain_incorrect"
      ) {
        show_data2 = show_data_start.filter(
          (row) => row.labeled == (show_labeled == "unlabeled" ? 0 : 1) //&& row.in_test == 1
        );
        if (show_labeled == "labeled") {
          console.log(
            "ignore_double_checked",
            urlParams.get("ignore_double_checked")
          );
          show_data2 = show_data2.filter(
            (row) =>
              row.double_checked == 0 ||
              row.label_undetermined.length > 0 ||
              urlParams.get("ignore_double_checked") == "on"
          );
        }
        const rank_modifier = urlParams.get("invert_ranking") == "on" ? -1 : +1;
        show_data2 = show_data2.sort(
          (a, b) =>
            rank_modifier * (a[as_ranking_column] - b[as_ranking_column])
        );
        show_data2 = show_data2.slice(
          0,
          show_labeled == "unlabeled" ? num_in_gallery - 1 : num_in_gallery
        );
        console.log("show_data2", show_data2);
      } else if (as_ranking_column == "high_label_entropy") {
        for (const row of data) {
          if (row.al_measure == 0 && row.in_test == 0) {
            show_data2.push(row);
          }
        }
      }
      if (show_labeled == "unlabeled") {
        for (const row of data) {
          if (row.in_test == 1 && row.labeled == 0) {
            show_data_test.push(row);
          }
        }
        console.log(show_data2.length, "show_data2");
        console.log(
          show_data_test.length,
          "show_data_test",
          Math.ceil(0.1 * show_data2.length)
        );
        show_data_test = _.sample(
          show_data_test,
          Math.ceil(0.1 * show_data2.length)
        );
        console.log(show_data_test.length, "show_data_test");
        if (as_ranking_column != "score_true") {
          for (const row of show_data_test) {
            console.log("test", row.uid);
            show_data2.push(row);
          }
        }
      }
      for (const row of show_data2) {
        d3.select("#dot-" + row.uid)
          .style("stroke", "#ff0000")
          .style("stroke-width", 2);
      }
      drawGallery(show_data2);
      console.log(show_data2.length, "show_data2");
    }
    //

    const drag = d3
      .drag()
      .filter(function filter(event) {
        return d3.event.ctrlKey && !d3.event.button;
      })
      .on("start", dragStart)
      .on("drag", dragMove)
      .on("end", function (event) {
        dragEnd(event, elementId);
      });

    d3.select(`#${svgId}`).call(drag);
  };

  this.render = function (data, x, y) {
    d3.select(`#${dotsId}`).remove();
    d3.select(`#${imagesId}`).remove();

    console.log("render", elementId, data.length);
    if (mode == "embedding") {
      dots = addDots(
        data,
        this.x,
        this.y,
        this.color_by,
        transform,
        width,
        height,
        d3Group,
        dotsId,
        this.filters
      );
      //
      console.log("embedding, images")
      this.images = addImages(
        data,
        this.x,
        this.y,
        transform,
        width,
        height,
        d3Group,
        imagesId,
        this.filters,    
        "e_0",
        "e_1",
        50,
        2 // in embedding space
      );
    } else if (mode == "tiles") {
      this.images = addImages(
        data,
        this.x,
        this.y,
        transform,
        width,
        height,
        d3Group,
        imagesId,
        this.filters,
        "patch_x",
        "patch_y"
      );
      console.log("blaat", this, this.images);
      console.log(
        "|images|",
        d3.select(`#${imagesId}`).selectAll("images").size()
      );
    }
  };
}

function zoomControl(elementId, transform, lastZoomUpdate, numUpdatesActive) {
  if (!transform) {
    return null;
  }

  const now = new Date().getTime() / 1000;
  if (now - lastZoomUpdate > 0.1) {
    if (numUpdatesActive == 0) {
      numUpdatesActive++;
      setTimeout(function () {
        console.log("zoomControl", elementId, gReferences.get(elementId));
        gReferences.get(elementId).render(data, x, y);
        lastZoomUpdate = now;
        numUpdatesActive--;
        console.log(
          "update took",
          new Date().getTime() / 1000 - now,
          numUpdatesActive
        );
      }, 0);
    }
  } else {
    console.log("too fast");
  }
}

const pointInPolygon = function (point, vs) {
  // console.log(point, vs);
  // ray-casting algorithm based on
  // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html/pnpoly.html

  var x = point[0],
    y = point[1];

  var inside = false;
  for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
    var xi = vs[i][0],
      yi = vs[i][1];
    var xj = vs[j][0],
      yj = vs[j][1];

    var intersect =
      yi > y != yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }

  return inside;
};
