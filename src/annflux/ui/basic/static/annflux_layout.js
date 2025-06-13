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
var config = {
  content: [
    {
      type: "row",
      content: [
        {
          type: "component",
          componentName: "Map",
          componentState: { label: "A" },
          width: "63",
        },
        {
          type: "column",
          content: [
            {
              type: "component",
              componentName: "Labels",
              componentState: { label: "B" },
              height: 10,
            },
            {
              type: "component",
              componentName: "Gallery",
              componentState: { label: "C" },
            },
            {
              type: "component",
              componentName: "Control",
              componentState: { label: "D" },
              height: 20,
            },
          ],
        },
      ],
    },
  ],
};

var labelConfig = {
  content: [
    {
      type: "row",
      content: [
        {
          type: "column",
          content: [
            {
              type: "component",
              componentName: "Map",
              componentState: { label: "A" },
              width: "30",
            },
            {
              type: "component",
              componentName: "Control",
              componentState: { label: "D" },
              height: 30,
            },
          ],
        },
        {
          type: "column",
          content: [
            {
              type: "component",
              componentName: "Labels",
              componentState: { label: "B" },
              height: 10,
            },
            {
              type: "component",
              componentName: "Gallery",
              componentState: { label: "C" },
            },
          ],
        },
      ],
    },
  ],
};

var partLabelConfig = {
  content: [
    {
      type: "row",
      content: [
        {
          type: "column",
          content: [
            {
              type: "component",
              componentName: "Map",
              componentState: { label: "A" },
              width: "30",
            },
            {
              type: "component",
              componentName: "Control",
              componentState: { label: "D" },
              height: 30,
            },
          ],
        },
        {
          type: "column",
          content: [
            {
              type: "component",
              componentName: "Labels",
              componentState: { label: "B" },
              height: 10,
            },
            {
              type: "component",
              componentName: "Gallery",
              componentState: { label: "C" },
            },
          ],
        },
        {
          type: "column",
          content: [
            {
              type: "component",
              componentName: "FullImage",
              componentState: { label: "E" },
            },
          ],
        },
      ],
    },
  ],
};
