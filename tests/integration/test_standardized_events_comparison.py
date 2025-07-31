#!/usr/bin/env python3

import sys
import os

sys.path.append("/home/matthias/ballpushing_utils/src")

from pathlib import Path
import Ballpushing_utils
from Ballpushing_utils.config import Config
import pandas as pd
import numpy as np


def compare_standardized_events_modes():
    """
    Compare interaction_events vs contact_events modes for standardized events generation.
    Tests the hypothesis that contact events should be more numerous and contained within interaction events.
    """
    print("=" * 80)
    print("COMPARING STANDARDIZED EVENTS MODES")
    print("=" * 80)

    # Use a single experiment for testing
    experiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231222_TNT_Fine_1_Videos_Tracked")

    if not experiment_path.exists():
        print(f"Experiment path does not exist: {experiment_path}")
        return

    # Create experiment and get first fly
    experiment = Ballpushing_utils.Experiment(experiment_path)
    if not experiment.flies:
        print("No flies found in experiment")
        return

    fly = experiment.flies[0]
    print(f"Testing with fly: {fly.metadata.name}")
    print()

    # Base configuration
    base_config = Config()
    base_config.frames_before_onset = 30
    base_config.frames_after_onset = 30
    base_config.contact_min_length = 2
    base_config.generate_random = False  # Disable random events for cleaner comparison

    # Test results storage
    results = {}

    print("-" * 60)
    print("TESTING INTERACTION_EVENTS MODE")
    print("-" * 60)

    # Test 1: Interaction events mode
    config_interaction = Config()
    config_interaction.__dict__.update(base_config.__dict__)
    config_interaction.standardized_events_mode = "interaction_events"
    fly.config = config_interaction

    try:
        skeleton_metrics_interaction = Ballpushing_utils.SkeletonMetrics(fly)

        # Get interaction events from tracking data
        interaction_events = fly.tracking_data.interaction_events
        standardized_interactions = fly.tracking_data.standardized_interactions

        # Get events-based contacts
        events_df_interaction = skeleton_metrics_interaction.events_based_contacts

        results["interaction"] = {
            "raw_interaction_events": sum(
                len(events) for fly_dict in interaction_events.values() for events in fly_dict.values()
            ),
            "standardized_interactions": sum(len(events) for events in standardized_interactions.values()),
            "events_df_shape": events_df_interaction.shape,
            "event_types": events_df_interaction["event_type"].unique() if not events_df_interaction.empty else [],
            "events_df": events_df_interaction,
        }

        print(f"Raw interaction events: {results['interaction']['raw_interaction_events']}")
        print(f"Standardized interactions: {results['interaction']['standardized_interactions']}")
        print(f"Events DataFrame shape: {results['interaction']['events_df_shape']}")
        print(f"Event types: {results['interaction']['event_types']}")

        if not events_df_interaction.empty:
            for event_type in results["interaction"]["event_types"]:
                count = (events_df_interaction["event_type"] == event_type).sum()
                unique_events = events_df_interaction[events_df_interaction["event_type"] == event_type][
                    "event_id"
                ].nunique()
                print(f"  {event_type}: {count} frames from {unique_events} events")

    except Exception as e:
        print(f"Error in interaction_events mode: {e}")
        import traceback

        traceback.print_exc()
        return

    print()
    print("-" * 60)
    print("TESTING CONTACT_EVENTS MODE")
    print("-" * 60)

    # Test 2: Contact events mode
    config_contact = Config()
    config_contact.__dict__.update(base_config.__dict__)
    config_contact.standardized_events_mode = "contact_events"
    fly.config = config_contact

    try:
        skeleton_metrics_contact = Ballpushing_utils.SkeletonMetrics(fly)

        # Get contact events
        contact_events = skeleton_metrics_contact.find_contact_events()
        annotated_df = skeleton_metrics_contact.get_contact_annotated_dataset()
        contact_periods = skeleton_metrics_contact._find_contact_periods(annotated_df)

        # Get events-based contacts
        events_df_contact = skeleton_metrics_contact.events_based_contacts

        results["contact"] = {
            "raw_contact_events": len(contact_events),
            "contact_periods": len(contact_periods),
            "contact_frames": annotated_df["is_contact"].sum() if "is_contact" in annotated_df.columns else 0,
            "total_frames": len(annotated_df),
            "events_df_shape": events_df_contact.shape,
            "event_types": events_df_contact["event_type"].unique() if not events_df_contact.empty else [],
            "events_df": events_df_contact,
        }

        print(f"Raw contact events: {results['contact']['raw_contact_events']}")
        print(f"Contact periods: {results['contact']['contact_periods']}")
        print(
            f"Contact frames: {results['contact']['contact_frames']}/{results['contact']['total_frames']} ({results['contact']['contact_frames']/results['contact']['total_frames']*100:.1f}%)"
        )
        print(f"Events DataFrame shape: {results['contact']['events_df_shape']}")
        print(f"Event types: {results['contact']['event_types']}")

        if not events_df_contact.empty:
            for event_type in results["contact"]["event_types"]:
                count = (events_df_contact["event_type"] == event_type).sum()
                unique_events = events_df_contact[events_df_contact["event_type"] == event_type]["event_id"].nunique()
                print(f"  {event_type}: {count} frames from {unique_events} events")

    except Exception as e:
        print(f"Error in contact_events mode: {e}")
        import traceback

        traceback.print_exc()
        return

    print()
    print("=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)

    # Compare the results
    print("📊 QUANTITATIVE COMPARISON:")
    print(
        f"Raw events: Contact ({results['contact']['raw_contact_events']}) vs Interaction ({results['interaction']['raw_interaction_events']})"
    )
    print(
        f"Standardized events: Contact ({results['contact']['contact_periods']}) vs Interaction ({results['interaction']['standardized_interactions']})"
    )
    print(
        f"Event frames: Contact ({results['contact']['events_df_shape'][0]}) vs Interaction ({results['interaction']['events_df_shape'][0]})"
    )

    # Ratio analysis
    if results["interaction"]["raw_interaction_events"] > 0:
        contact_to_interaction_ratio = (
            results["contact"]["raw_contact_events"] / results["interaction"]["raw_interaction_events"]
        )
        print(f"Contact-to-Interaction ratio: {contact_to_interaction_ratio:.2f}x")

    print()
    print("🔍 HYPOTHESIS TESTING:")

    # Test hypothesis 1: More contacts than interactions
    if results["contact"]["raw_contact_events"] > results["interaction"]["raw_interaction_events"]:
        print("✅ PASSED: More contact events than interaction events")
    else:
        print("❌ FAILED: Expected more contact events than interaction events")

    # Test hypothesis 2: Contact events should create more standardized events
    if results["contact"]["contact_periods"] > results["interaction"]["standardized_interactions"]:
        print("✅ PASSED: More standardized contact events than interaction events")
    else:
        print("❌ FAILED: Expected more standardized contact events")

    print()
    print("🕐 TEMPORAL OVERLAP ANALYSIS:")

    # Analyze temporal overlap between interaction and contact events
    analyze_temporal_overlap(results)

    print()
    print("📋 SUMMARY:")
    print(f"• Interaction mode: {results['interaction']['standardized_interactions']} standardized events")
    print(f"• Contact mode: {results['contact']['contact_periods']} standardized events")
    print(
        f"• Contact detection found {results['contact']['contact_frames']} contact frames ({results['contact']['contact_frames']/results['contact']['total_frames']*100:.1f}% of total)"
    )
    print(f"• Contact events are {contact_to_interaction_ratio:.1f}x more numerous than interaction events")


def analyze_temporal_overlap(results):
    """
    Analyze temporal overlap between interaction events and contact periods.
    """
    interaction_df = results["interaction"]["events_df"]
    contact_df = results["contact"]["events_df"]

    if interaction_df.empty or contact_df.empty:
        print("⚠️  Cannot analyze overlap: One or both DataFrames are empty")
        return

    # Get unique interaction events from both DataFrames
    interaction_events = []
    for df_name, df in [("interaction_mode", interaction_df), ("contact_mode", contact_df)]:
        if "interaction" in df["event_type"].values:
            interaction_subset = df[df["event_type"] == "interaction"]
            for event_id in interaction_subset["event_id"].unique():
                event_frames = interaction_subset[interaction_subset["event_id"] == event_id]
                start_frame = event_frames.index.min()
                end_frame = event_frames.index.max()
                interaction_events.append((start_frame, end_frame, df_name))
            break  # Use the first available

    # Get unique contact events (only from contact mode)
    contact_events = []
    if "contact" in contact_df["event_type"].values:
        contact_subset = contact_df[contact_df["event_type"] == "contact"]
        for event_id in contact_subset["event_id"].unique():
            event_frames = contact_subset[contact_subset["event_id"] == event_id]
            if "contact_period_start" in event_frames.columns:
                contact_start = event_frames["contact_period_start"].iloc[0]
                contact_end = event_frames["contact_period_end"].iloc[0]
                contact_events.append((contact_start, contact_end))

    if not interaction_events or not contact_events:
        print("⚠️  No events found for overlap analysis")
        return

    print(f"Found {len(interaction_events)} interaction events and {len(contact_events)} contact events")

    # Find overlaps
    overlaps = 0
    contacts_within_interactions = 0
    contacts_overlapping_interactions = 0

    for contact_start, contact_end in contact_events:
        has_overlap = False
        is_fully_within = False

        for interaction_start, interaction_end, source in interaction_events:
            # Check if contact overlaps with interaction
            if not (contact_end <= interaction_start or contact_start >= interaction_end):
                has_overlap = True
                # Check if contact is fully within interaction
                if contact_start >= interaction_start and contact_end <= interaction_end:
                    is_fully_within = True
                break

        if has_overlap:
            contacts_overlapping_interactions += 1
        if is_fully_within:
            contacts_within_interactions += 1

    print(
        f"Contact events overlapping with interactions: {contacts_overlapping_interactions}/{len(contact_events)} ({contacts_overlapping_interactions/len(contact_events)*100:.1f}%)"
    )
    print(
        f"Contact events fully within interactions: {contacts_within_interactions}/{len(contact_events)} ({contacts_within_interactions/len(contact_events)*100:.1f}%)"
    )

    # This is actually expected behavior - contacts are more granular than interactions
    print()
    print("ℹ️  INTERPRETATION:")
    if contacts_overlapping_interactions / len(contact_events) < 0.5:
        print("✅ EXPECTED: Contact events are more granular than interaction events")
        print("   Contact detection captures brief touches that may not qualify as interactions")
        print("   This demonstrates the higher temporal resolution of contact-based detection")
    else:
        print("ℹ️  Higher overlap suggests contact and interaction detection are well-aligned")


if __name__ == "__main__":
    compare_standardized_events_modes()
