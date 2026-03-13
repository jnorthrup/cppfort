#include <iostream>
#include <string>
#include <string_view>

#include "rbcursive.cpp"

int main() {
    {
        scan_session session{};
        auto result = project_to_feature_stream(std::string_view{"  chart"}, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: project_to_feature_stream did not accept chart head\n";
            return 1;
        }
        if (!result.value.has_value() || result.value.value() != std::string_view{"chart"}) {
            std::cerr << "FAIL: project_to_feature_stream returned wrong head value\n";
            return 2;
        }
        if (result.consumed != 7) {
            std::cerr << "FAIL: project_to_feature_stream consumed " << result.consumed << " bytes, expected 7\n";
            return 3;
        }
        if (session.features.size() != 1) {
            std::cerr << "FAIL: expected 1 feature from chart head, got " << session.features.size() << "\n";
            return 4;
        }
        if (!(session.features[0].kind == feature_kind::keyword)) {
            std::cerr << "FAIL: expected keyword feature for chart head\n";
            return 5;
        }
    }

    {
        constexpr std::string_view source = " chart(identity)";
        scan_session session{};
        auto parser = pure2_keyword_group("chart");
        auto result = parser(source, 0, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: pure2_keyword_group did not accept balanced group\n";
            return 6;
        }
        if (!result.value.has_value() || result.value.value() != std::string_view{"(identity)"}) {
            std::cerr << "FAIL: pure2_keyword_group returned wrong group payload\n";
            return 7;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: pure2_keyword_group did not consume full source\n";
            return 8;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: expected keyword + group features, got " << session.features.size() << "\n";
            return 9;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::group)) {
            std::cerr << "FAIL: feature sequence was not keyword -> group\n";
            return 10;
        }
    }

    {
        constexpr std::string_view source = " chart identity(point: f64)";
        scan_session session{};
        auto result = project_chart_declaration_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: chart declaration surface did not accept a complete signature\n";
            return 11;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: chart declaration surface returned the wrong matched slice\n";
            return 12;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: chart declaration surface did not consume the full signature\n";
            return 13;
        }
        if (session.features.size() != 3) {
            std::cerr << "FAIL: expected keyword + identifier + group features, got "
                      << session.features.size() << "\n";
            return 14;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier &&
              session.features[2].kind == feature_kind::group)) {
            std::cerr << "FAIL: chart declaration feature sequence was not keyword -> identifier -> group\n";
            return 15;
        }
        if (session.features[2].semantic != std::string{"chart_parameters"}) {
            std::cerr << "FAIL: chart declaration parameter group semantic label mismatch\n";
            return 16;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete chart declaration emitted unexpected diagnostics\n";
            return 17;
        }
    }

    {
        constexpr std::string_view source = " chart identity(";
        scan_session session{};
        auto result = project_chart_declaration_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete chart declaration did not request more input\n";
            return 18;
        }
        if (result.message != std::string{"chart parameter group incomplete"}) {
            std::cerr << "FAIL: incomplete chart declaration returned wrong message: "
                      << result.message << "\n";
            return 19;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: incomplete chart declaration should have recorded keyword + identifier before stalling\n";
            return 20;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete chart declaration should emit exactly one diagnostic\n";
            return 21;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete chart declaration diagnostic severity was not incomplete\n";
            return 22;
        }
        if (session.diagnostics[0].message != std::string{"chart parameter group incomplete"}) {
            std::cerr << "FAIL: incomplete chart declaration diagnostic message mismatch\n";
            return 23;
        }
    }

    {
        constexpr std::string_view source = " chart (point: f64)";
        scan_session session{};
        auto result = project_chart_declaration_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: chart declaration without a name did not reject\n";
            return 24;
        }
        if (result.message != std::string{"chart name expected"}) {
            std::cerr << "FAIL: chart declaration without a name returned wrong message: "
                      << result.message << "\n";
            return 25;
        }
        if (session.features.size() != 1 || !(session.features[0].kind == feature_kind::keyword)) {
            std::cerr << "FAIL: rejected chart declaration should only preserve the leading keyword feature\n";
            return 26;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected chart declaration should emit exactly one diagnostic\n";
            return 27;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected chart declaration diagnostic severity was not error\n";
            return 28;
        }
        if (session.diagnostics[0].message != std::string{"chart name expected"}) {
            std::cerr << "FAIL: rejected chart declaration diagnostic message mismatch\n";
            return 29;
        }
    }

    {
        constexpr std::string_view source = " atlas[shifted, identity]";
        scan_session session{};
        auto result = project_atlas_literal_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: atlas literal surface did not accept a complete atlas literal\n";
            return 30;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: atlas literal surface returned the wrong matched slice\n";
            return 31;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: atlas literal surface did not consume the full literal\n";
            return 32;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: expected atlas keyword + bracket group features, got "
                      << session.features.size() << "\n";
            return 33;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::group)) {
            std::cerr << "FAIL: atlas literal feature sequence was not keyword -> group\n";
            return 34;
        }
        if (session.features[1].semantic != std::string{"atlas_elements"}) {
            std::cerr << "FAIL: atlas literal group semantic label mismatch\n";
            return 35;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete atlas literal emitted unexpected diagnostics\n";
            return 36;
        }
    }

    {
        constexpr std::string_view source = " atlas[shifted, identity";
        scan_session session{};
        auto result = project_atlas_literal_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete atlas literal did not request more input\n";
            return 37;
        }
        if (result.message != std::string{"atlas element group incomplete"}) {
            std::cerr << "FAIL: incomplete atlas literal returned wrong message: "
                      << result.message << "\n";
            return 38;
        }
        if (session.features.size() != 1 || !(session.features[0].kind == feature_kind::keyword)) {
            std::cerr << "FAIL: incomplete atlas literal should only preserve the leading keyword feature\n";
            return 39;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete atlas literal should emit exactly one diagnostic\n";
            return 40;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete atlas literal diagnostic severity was not incomplete\n";
            return 41;
        }
        if (session.diagnostics[0].message != std::string{"atlas element group incomplete"}) {
            std::cerr << "FAIL: incomplete atlas literal diagnostic message mismatch\n";
            return 42;
        }
    }

    {
        constexpr std::string_view source = " atlas shifted";
        scan_session session{};
        auto result = project_atlas_literal_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: atlas literal without brackets did not reject\n";
            return 43;
        }
        if (result.message != std::string{"atlas element group expected"}) {
            std::cerr << "FAIL: atlas literal without brackets returned wrong message: "
                      << result.message << "\n";
            return 44;
        }
        if (session.features.size() != 1 || !(session.features[0].kind == feature_kind::keyword)) {
            std::cerr << "FAIL: rejected atlas literal should only preserve the leading keyword feature\n";
            return 45;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected atlas literal should emit exactly one diagnostic\n";
            return 46;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected atlas literal diagnostic severity was not error\n";
            return 47;
        }
        if (session.diagnostics[0].message != std::string{"atlas element group expected"}) {
            std::cerr << "FAIL: rejected atlas literal diagnostic message mismatch\n";
            return 48;
        }
    }

    {
        constexpr std::string_view source =
            " chart identity(point: f64) {\n"
            "  contains point <= 20.0\n"
            "  project -> coords[point]\n"
            "  embed(local) -> local[0]\n"
            "}";
        scan_session session{};
        auto result = project_chart_definition_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: chart definition surface did not accept a complete definition\n";
            return 49;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: chart definition surface returned the wrong matched slice\n";
            return 50;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: chart definition surface did not consume the full definition\n";
            return 51;
        }
        if (session.features.size() != 12) {
            std::cerr << "FAIL: expected chart head + contains/project/embed body features, got "
                      << session.features.size() << "\n";
            return 52;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier &&
              session.features[2].kind == feature_kind::group &&
              session.features[3].kind == feature_kind::keyword &&
              session.features[4].kind == feature_kind::surface &&
              session.features[5].kind == feature_kind::keyword &&
              session.features[6].kind == feature_kind::keyword &&
              session.features[7].kind == feature_kind::group &&
              session.features[8].kind == feature_kind::keyword &&
              session.features[9].kind == feature_kind::group &&
              session.features[10].kind == feature_kind::surface &&
              session.features[11].kind == feature_kind::group)) {
            std::cerr << "FAIL: chart definition feature sequence did not preserve head + contains/project/embed body order\n";
            return 53;
        }
        if (session.features[2].semantic != std::string{"chart_parameters"}) {
            std::cerr << "FAIL: chart definition parameter group semantic label mismatch\n";
            return 54;
        }
        if (session.features[4].semantic != std::string{"chart_contains_expression"}) {
            std::cerr << "FAIL: chart definition contains expression semantic label mismatch\n";
            return 55;
        }
        if (session.features[6].semantic != std::string{"coords"}) {
            std::cerr << "FAIL: chart definition coords keyword semantic label mismatch\n";
            return 56;
        }
        if (session.features[7].semantic != std::string{"coords_elements"}) {
            std::cerr << "FAIL: chart definition coords element group semantic label mismatch\n";
            return 57;
        }
        if (session.features[9].semantic != std::string{"chart_embed_parameters"}) {
            std::cerr << "FAIL: chart definition embed parameter group semantic label mismatch\n";
            return 58;
        }
        if (session.features[10].semantic != std::string{"chart_embed_expression"}) {
            std::cerr << "FAIL: chart definition embed expression semantic label mismatch\n";
            return 59;
        }
        if (session.features[11].semantic != std::string{"chart_body"}) {
            std::cerr << "FAIL: chart definition body group semantic label mismatch\n";
            return 60;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete chart definition emitted unexpected diagnostics\n";
            return 61;
        }
    }

    {
        constexpr std::string_view source = " chart identity(point: f64) { project -> coords[point] ";
        scan_session session{};
        auto result = project_chart_definition_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete chart definition body did not request more input\n";
            return 57;
        }
        if (result.message != std::string{"chart body incomplete"}) {
            std::cerr << "FAIL: incomplete chart definition body returned wrong message: "
                      << result.message << "\n";
            return 61;
        }
        if (session.features.size() != 6) {
            std::cerr << "FAIL: incomplete chart definition should preserve head + project body features before stalling\n";
            return 62;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete chart definition should emit exactly one diagnostic\n";
            return 63;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete chart definition diagnostic severity was not incomplete\n";
            return 64;
        }
        if (session.diagnostics[0].message != std::string{"chart body incomplete"}) {
            std::cerr << "FAIL: incomplete chart definition diagnostic message mismatch\n";
            return 65;
        }
    }

    {
        constexpr std::string_view source = " chart identity(point: f64) project -> coords[point] }";
        scan_session session{};
        auto result = project_chart_definition_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: chart definition without a body opener did not reject\n";
            return 66;
        }
        if (result.message != std::string{"chart body expected"}) {
            std::cerr << "FAIL: chart definition without a body opener returned wrong message: "
                      << result.message << "\n";
            return 67;
        }
        if (session.features.size() != 3) {
            std::cerr << "FAIL: rejected chart definition should preserve keyword + name + parameter group features\n";
            return 68;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected chart definition should emit exactly one diagnostic\n";
            return 69;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected chart definition diagnostic severity was not error\n";
            return 70;
        }
        if (session.diagnostics[0].message != std::string{"chart body expected"}) {
            std::cerr << "FAIL: rejected chart definition diagnostic message mismatch\n";
            return 71;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  contains point <= 20.0\n"
            "  project -> coords[point - 10.0]\n"
            "  embed(local) -> local[0] + 10.0\n"
            "}";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: chart body surface did not accept repo-real project/embed clauses\n";
            return 69;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: chart body surface returned the wrong matched slice\n";
            return 70;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: chart body surface did not consume the full body\n";
            return 71;
        }
        if (session.features.size() != 9) {
            std::cerr << "FAIL: expected contains + project + embed clause features plus body group, got "
                      << session.features.size() << "\n";
            return 72;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::surface &&
              session.features[2].kind == feature_kind::keyword &&
              session.features[3].kind == feature_kind::keyword &&
              session.features[4].kind == feature_kind::group &&
              session.features[5].kind == feature_kind::keyword &&
              session.features[6].kind == feature_kind::group &&
              session.features[7].kind == feature_kind::surface &&
              session.features[8].kind == feature_kind::group)) {
            std::cerr << "FAIL: chart body feature sequence did not preserve clause-first projection order\n";
            return 73;
        }
        if (session.features[1].semantic != std::string{"chart_contains_expression"}) {
            std::cerr << "FAIL: chart contains expression semantic label mismatch\n";
            return 74;
        }
        if (session.features[3].semantic != std::string{"coords"}) {
            std::cerr << "FAIL: chart project coords keyword semantic label mismatch\n";
            return 75;
        }
        if (session.features[4].semantic != std::string{"coords_elements"}) {
            std::cerr << "FAIL: chart project coords element group semantic label mismatch\n";
            return 76;
        }
        if (session.features[6].semantic != std::string{"chart_embed_parameters"}) {
            std::cerr << "FAIL: chart embed parameter group semantic label mismatch\n";
            return 77;
        }
        if (session.features[7].semantic != std::string{"chart_embed_expression"}) {
            std::cerr << "FAIL: chart embed expression semantic label mismatch\n";
            return 78;
        }
        if (session.features[8].semantic != std::string{"chart_body"}) {
            std::cerr << "FAIL: chart body group semantic label mismatch\n";
            return 79;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete chart body emitted unexpected diagnostics\n";
            return 80;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  project -> coords[point\n"
            "}";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete coords literal did not request more input\n";
            return 80;
        }
        if (result.message != std::string{"coords element group incomplete"}) {
            std::cerr << "FAIL: incomplete coords literal returned wrong message: "
                      << result.message << "\n";
            return 81;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: incomplete coords literal should preserve project + coords keyword features\n";
            return 82;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete coords literal should emit exactly one diagnostic\n";
            return 83;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete coords literal diagnostic severity was not incomplete\n";
            return 84;
        }
        if (session.diagnostics[0].message != std::string{"coords element group incomplete"}) {
            std::cerr << "FAIL: incomplete coords literal diagnostic message mismatch\n";
            return 85;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  project -> coords[point]\n"
            "  embed(local) -> ";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete embed clause did not request more input\n";
            return 86;
        }
        if (result.message != std::string{"embed expression expected"}) {
            std::cerr << "FAIL: incomplete embed clause returned wrong message: "
                      << result.message << "\n";
            return 87;
        }
        if (session.features.size() != 5) {
            std::cerr << "FAIL: incomplete chart body should preserve project clause and embed head features\n";
            return 88;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete chart body should emit exactly one diagnostic\n";
            return 89;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete chart body diagnostic severity was not incomplete\n";
            return 90;
        }
        if (session.diagnostics[0].message != std::string{"embed expression expected"}) {
            std::cerr << "FAIL: incomplete chart body diagnostic message mismatch\n";
            return 91;
        }
    }

    {
        constexpr std::string_view source = "{ project coords[point] }";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: chart body without clause arrow did not reject\n";
            return 92;
        }
        if (result.message != std::string{"project arrow expected"}) {
            std::cerr << "FAIL: chart body without clause arrow returned wrong message: "
                      << result.message << "\n";
            return 93;
        }
        if (session.features.size() != 1 || !(session.features[0].kind == feature_kind::keyword)) {
            std::cerr << "FAIL: rejected chart body should only preserve the project keyword feature\n";
            return 94;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected chart body should emit exactly one diagnostic\n";
            return 95;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected chart body diagnostic severity was not error\n";
            return 96;
        }
        if (session.diagnostics[0].message != std::string{"project arrow expected"}) {
            std::cerr << "FAIL: rejected chart body diagnostic message mismatch\n";
            return 97;
        }
    }

    std::cout << "selfhost rbcursive smoke passed\n";
    return 0;
}
