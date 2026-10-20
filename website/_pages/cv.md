---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

**Gabriel Casciano, B.Eng., E.I.T.**<br>
Mechatronics, Sensors & Embedded Systems Engineer<br>
[gabecasciano@gmail.com](mailto:gabecasciano@gmail.com) · [linkedin.com/in/gabe-casciano-52a717230](https://www.linkedin.com/in/gabe-casciano-52a717230)

Mechatronics and embedded systems engineer (E.I.T.) with 5+ years of industry and applied-research experience taking sensor and robotic systems from concept to field deployment: mechanical design, mixed-signal PCB design, real-time firmware, motor control, sensor calibration, computer vision, and edge ML on NVIDIA Jetson. Designed and flew a 2-axis ISR tracking gimbal, built an ASIC-based acoustic drone detector, and delivered grant-funded sensing systems into steel mills and manufacturing plants. Experienced grant writer, project manager, and mentor to co-op engineers.

## Education

* **MASc, Systems Design Engineering**, University of Waterloo, expected Winter 2028
  * Vision and Image Processing (VIP) Lab, supervised by Prof. John Zelek
* **B.Eng., Computer Engineering**, Toronto Metropolitan University (formerly Ryerson), 2016 - 2021
  * Embedded hardware and firmware (FPGA, SoC, MCU/MPU), operating systems and RTOS, networking, control systems and system modelling, mixed-signal design and DSP

## Professional Experience

### TensorOne Inc.
*Mechatronics Engineer & Engineering Operations Manager · Kitchener-Waterloo, ON · Apr 2026 - Present*

Responsible for TensorOne's intelligence, surveillance and reconnaissance (ISR) payloads: mechanics, electronics, firmware, control, and onboard vision, plus the procurement and supply chain behind prototype builds.

* **2-Axis ISR Tracking Gimbal (flight-proven).** Owned the system end to end. Tracks targets out to 100 m and objects moving up to 5 m/s at 20 m.
  * Mechanical: Onshape mechanism around GM4108 BLDC motors, GT2 belt reduction on yaw, and a hollow-bore pulley routing USB 3.1 and CAN FD through the rotating joint. Custom FeatureScripts for parametric pulleys and planetary gearboxes.
  * Electronics: 4-layer STM32G474 FOC motor controller (DRV8311, CAN FD, USB-C, 4S battery monitoring), STM32G431 IMU node (LSM6DSV, MMC5983, MCP2515 CAN), and a USB-C AS5048A magnetic encoder board.
  * Calibration and control: IMU bias/scale and magnetometer hard/soft-iron calibration; FOC and PID tuning closed on encoder, IMU, and vision feedback.
* **Vision, Edge AI & Payload Integration.** Brought up a Framos IMX900 global-shutter camera on a Jetson Orin Nano Super (Linux driver, device tree, GStreamer pipeline, intrinsic/extrinsic calibration) alongside an IMX636 event camera. Trained, tuned, and deployed RF-DETR (frame) and RVT (event) detectors on the Jetson, feeding the gimbal controller for real-time on-board tracking.
* **Commercial Gimbal Integration & Autonomous Search-and-Track.** Implemented the Skydroid C14 Ethernet/UDP control protocol, integrated it with ArduPilot (MAVLink) and the Jetson, added a laser rangefinder, and wrote the autonomous search-and-track state machine.
* **Acoustic Drone Detection.** Prototype hardware around the Northern Mechatronics NM180100 ASIC with a calibrated 48 kHz microphone channel, and a spectrogram + neural-network pipeline that detects 7-inch drones at 30 m AGL.
* **Engineering Operations.** Hardware procurement and supplier sourcing, including international purchasing, Incoterms, and customs import; coordination of PCB fabrication and assembly.

### Mohawk College - IdeaWorks, IIoT & Sensor Systems Lab
*Lead Sensors & Systems Engineer · Hamilton, ON · Jan 2022 - Mar 2026*

* Led the lab's sensor and mechatronic systems, delivering 4-5 applied-research projects a year for partners including ArcelorMittal Dofasco, Stelco, Motsai, and TFT Global. Built a test fixture or calibration rig for every project.
* Met prospective partners, wrote scopes of work, and wrote and won about 5 applied-research grants a year (NSERC ARD, ARTP and Engage; SONAMI; OCI), with more than $1M in managed funding.
* Built budgets and work breakdowns, introduced Agile sprints and stand-ups, ran weekly student and partner meetings, and coordinated on-site testing in production environments.
* Hired, trained, and mentored 3-4 co-op students a year (20+ in total) with hands-on challenges covering Git, Python, Qt, Docker, MQTT, OpenCV, and embedded tooling.
* Built Python IoT back ends (FastAPI, SQLAlchemy, Pydantic, PostgreSQL) over MQTT/HTTP, Qt operator applications, and Raspberry Pi sensor gateways; administered Debian/RHEL servers; delivered SOPs and implementation notes at project close.

**Year 1 (2022)**

* **Slag Raking Vision.** Emissivity-based segmentation of slag over a ~300-tonne ladle of 1,600 °C liquid steel using a FLIR A500 thermal camera, with a Qt HMI showing slag-coverage estimates. Took over the project, re-planned and budgeted the remaining phases, selected the final camera system, and trained the student team. Installed in the Ladle Metallurgy Facility control room and used by operators to guide raking; estimated savings to the partner of ~$150k.
* **All-in-One Industrial Sensor.** Stackable, reconfigurable predictive-maintenance sensor on Zephyr RTOS. Refactored the schematics, completed the Altium layout and DRC/ERC, produced fab/assembly packages, procured PCBA, wrote the firmware, and ran all electrical and functional testing.
* **Industrial Predictive Maintenance System.** Characterized a scissor lift for a digital twin and failure-prediction model using motor/gearbox temperature and vibration plus motor V/I, flagging 4-6σ anomalies to a mobile operator app. Managed the project, wrote the acquisition and analysis firmware, led a student dashboard team, and presented results to the partner.
* **Mouvion IIoT Sensor Platform.** Postage-stamp-sized motion sensor with Motsai integrating BLE, Wi-Fi, and LoRa. Selected components, designed the schematic and PCB layout, wrote the low-level firmware, and validated the design jointly with Motsai.

**Year 2 (2023)**

* **Blast Furnace Tuyere Camera Vision.** Machine-vision system that detects scabs at the tuyeres, logs events, and runs on a Dockerized microservice back end scalable to all 24 tuyeres. Wrote the plan and won a multi-year NSERC ARD grant, ran discovery, designed the camera system, collected the dataset, and led two student cohorts through multiple production deployments. The partner pursued a fee-for-service build-out.
* **Conveyor Belt Monitoring.** Millimetre-resolution point-cloud defect detection for a ~1.6 km steel-reinforced conveyor feeding a blast furnace. Won a multi-year NSERC ARTP grant, selected a high-speed laser-line scanner, and built a scaled lab conveyor where the proof of concept was demonstrated.
* **3D Warehouse Digital Twin.** Unity digital twin linked to a live warehouse management database showing steel-coil inventory and crane positions. Won a SONAMI grant, co-designed the architecture, and hired and managed two students.
* **Wireless Ergonomic Tool-Sensing System.** Hand-tool sensor with six FSR grip-force channels and a 6-DOF IMU. Won a SONAMI grant and designed dual-MCU boards in Altium (STM32F446RE with an op-amp FSR front end, IMU, and SD logging; ESP32-C6 for Wi-Fi/MQTT and the operator display). Calibrated force against reference masses and delivered the firmware; the partner piloted it on the production floor.

**Year 3 (2024)**

* **Iris SEN.** Servo-actuated iris that models submerged-entry-nozzle clogging in a full-scale water model, with Mohawk's AMIC and ArcelorMittal Dofasco. Guided the design, instrumented the motors and encoders, deployed it on-site, and trained a student to build the Python SDK and Qt control application.
* **LoRa Remote-Sensing Network.** Proof-of-concept LoRa P2P network for remote sensing in the Northwest Territories, with power profiling alongside Mohawk's EPIC centre.

**Year 4 (2025 - Mar 2026)**

* **Continuous-Casting Water Model Control & Sensor System.** Won the NSERC ARD proposal. Instrumented the model with anemometers and ultrasonic level sensors on an NI cDAQ, with an Arduino Opta PLC for safety-critical I/O, publishing over MQTT to a database logger and Qt operator app. Built an ultrasonic calibration jig and used both sensors as ground truth for camera-based level, surface-waveform, and optical-flow measurement. The data validates the partner's CFD models.
* **Dynamic Pick-to-Light System.** Secured an 18-month OCI C2C grant, hired and trained the student developer, and led implementation of a picklist-driven light-directed picking system.
* **Tote-Sorting Mobile Robot (proposal).** Led the literature review, concept (collaborative robot on an OTTO 600 base), and NSERC ARD proposal.

## Grants & Funding

| Program | Project | Partner | Funding |
|---|---|---|---|
| NSERC ARD | Blast Furnace Tuyere Camera Vision | Stelco | $430k total ($300k NSERC + $130k partner cash), multi-year |
| NSERC ARTP | Conveyor Belt Monitoring | Stelco | $136k total ($96k NSERC + $40k partner cash), multi-year |
| SONAMI | 3D Warehouse Digital Twin | CareGo Tek | $20k total ($15k SONAMI + $5k partner cash) |
| SONAMI | Wireless Ergonomic Tool-Sensing System | Industry partner | 6 months, 2 students + materials |
| OCI C2C | Dynamic Pick-to-Light System | Industry partner | 18 months |
| NSERC ARD | Continuous-Casting Water Model & Iris SEN | ArcelorMittal Dofasco | Multi-year |
| NSERC ARD | Tote-Sorting Mobile Robot | Industry partner | Proposal author |
| NSERC ARD | Slag Raking Vision (led and delivered) | ArcelorMittal Dofasco | $300k NSERC + $120k cash + $210k in-kind |
| NSERC ARTP + Engage | Industrial Predictive Maintenance (led and delivered) | Handling Specialty | $36k total |
| NSERC ARTP + Engage | Mouvion IIoT Sensor Platform (delivered) | Motsai | - |

## Teaching & Mentorship

* **Part-Time Faculty, Electrical Engineering Technology**, Mohawk College, Winter 2024
  * Developed and delivered a seminar-style introductory AutoCAD course covering drafting fundamentals, notation, and electrical schematic drafting.
* **Applied Research & Development Course (capstone alternative)**, Mohawk College, 2024
  * Mentored a student section building pick-to-light systems from COTS parts and Arduino, benchmarked against a commercial system. Every student delivered a complete working system.
* **Lead Instructor, Coach & Course Coordinator**, Dr. X Academy, Richmond Hill, ON, Aug 2018 - Dec 2021
  * Coached teams of 20-50 students in grades 1-12 in FLL, FTC, FRC, VEX IQ, VEX EDR, and WRO, winning VEX IQ World Championship titles, FLL and WRO championships, and an FTC provincial title.
  * Managed the instructor team and designed STEAM curriculum in robotics and control, electronics and Arduino, 3D design and printing, programming, vision, Unity, and app development, some of it licensed to other STEM schools.
  * Mentored private students through self-directed projects such as human-detection systems, autonomous vehicles and ROVs, and IoT sensors.

## Robotics Competitions & Leadership

### Ryerson Rams Robotics (R3)
*Controls & Electronics Lead; Team Captain (2019 - 2021) · Toronto, ON · Sep 2016 - May 2021*

* **University Rover Challenge (2017 - 2021):** 2nd overall in 2019 in the Utah Mars-analogue desert. Designed the rover's control system and embedded critical I/O, and managed the hardware and software teams.
* **RoboNation RoboSub (2019):** 17th overall with an autonomous submarine. Led the controls team, designed the power distribution, thruster/solenoid, and telemetry boards, and tuned the 6-DOF feedback control.
* **Ramsbot (2017):** designed the control and vision systems for a basketball-shooting robot for a Toronto Raptors (MLSE) halftime demo, making ~90% of shots inside the 3-point line.
* **Canadian International Rover Challenge (2018 - 2021).**
* STM32 control electronics and firmware with FIR/IIR and Kalman/EKF filtering on ARM DSP and feed-forward/anti-windup PID.
* Grew the team from about 50 to 120 members and from one to 2-3 projects a year; built training modules, managed budgets up to $60k, and met with sponsors.

## Technical Skills

* **Sensors & Instrumentation:** IMU/magnetometer, force (FSR, load cell), ultrasonic level, anemometry, thermal (FLIR), event and global-shutter cameras, laser-line scanners, microphones, NI cDAQ, calibration jigs and test fixtures
* **Electronics & PCB:** Altium Designer, KiCad; analog front ends, BLDC/FOC power stages, 4-layer stackups, differential-pair routing (CAN FD, USB), multi-radio modules (BLE, Wi-Fi, LoRa), DFM, bring-up and rework
* **Embedded:** C/C++, Rust; STM32 HAL, ESP-IDF, nRF, AVR/SAMD, Zephyr, FreeRTOS, CMSIS; FOC, PID; FIR/IIR, Kalman/EKF
* **Vision & ML:** Linux camera drivers and device tree, GStreamer, OpenCV, RF-DETR, RVT, PyTorch, TensorFlow; NVIDIA Jetson Orin; camera calibration
* **Interfaces:** CAN/CAN FD, SPI, I2C, I3C, UART, USB, RS-485, MIPI CSI-2, Ethernet/UDP, MQTT, Modbus, OPC UA, LoRa, BLE, MAVLink/ArduPilot
* **Software & Linux:** Python (NumPy, pandas, FastAPI, SQLAlchemy, Pydantic, PyQt), Qt, Unity; PostgreSQL; Debian/RHEL, systemd, Docker, Git
* **Mechanical:** Onshape (incl. custom FeatureScripts), SolidWorks, Fusion 360, AutoCAD; belt and gear transmissions, 3D printing, CNC and manual machining
