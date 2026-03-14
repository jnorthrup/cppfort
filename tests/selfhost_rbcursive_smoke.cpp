#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include "rbcursive.cpp"

#ifndef CPPFORT_SOURCE_DIR
#define CPPFORT_SOURCE_DIR "."
#endif

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
        if (session.features.size() != 4) {
            std::cerr << "FAIL: expected atlas keyword + identifier elements + bracket group, got "
                      << session.features.size() << "\n";
            return 33;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier &&
              session.features[2].kind == feature_kind::identifier &&
              session.features[3].kind == feature_kind::group)) {
            std::cerr << "FAIL: atlas literal feature sequence was not keyword -> identifier -> identifier -> group\n";
            return 34;
        }
        if (session.features[1].semantic != std::string{"shifted"}) {
            std::cerr << "FAIL: atlas literal first identifier semantic label mismatch\n";
            return 35;
        }
        if (session.features[2].semantic != std::string{"identity"}) {
            std::cerr << "FAIL: atlas literal second identifier semantic label mismatch\n";
            return 36;
        }
        if (session.features[3].semantic != std::string{"atlas_elements"}) {
            std::cerr << "FAIL: atlas literal group semantic label mismatch\n";
            return 37;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete atlas literal emitted unexpected diagnostics\n";
            return 38;
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
            return 39;
        }
        if (session.features.size() != 3) {
            std::cerr << "FAIL: incomplete atlas literal should preserve the keyword and completed identifier elements\n";
            return 40;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier &&
              session.features[2].kind == feature_kind::identifier)) {
            std::cerr << "FAIL: incomplete atlas literal did not preserve keyword -> identifier -> identifier ordering\n";
            return 41;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete atlas literal should emit exactly one diagnostic\n";
            return 42;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete atlas literal diagnostic severity was not incomplete\n";
            return 43;
        }
        if (session.diagnostics[0].message != std::string{"atlas element group incomplete"}) {
            std::cerr << "FAIL: incomplete atlas literal diagnostic message mismatch\n";
            return 44;
        }
    }

    {
        constexpr std::string_view source = " atlas[shifted identity]";
        scan_session session{};
        auto result = project_atlas_literal_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: atlas literal without an element separator did not reject\n";
            return 45;
        }
        if (result.message != std::string{"atlas element separator expected"}) {
            std::cerr << "FAIL: atlas literal without an element separator returned wrong message: "
                      << result.message << "\n";
            return 46;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: rejected atlas literal should preserve the leading keyword and first identifier\n";
            return 47;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier)) {
            std::cerr << "FAIL: rejected atlas literal did not preserve keyword -> identifier ordering\n";
            return 48;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected atlas separator error should emit exactly one diagnostic\n";
            return 49;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected atlas separator diagnostic severity was not error\n";
            return 50;
        }
        if (session.diagnostics[0].message != std::string{"atlas element separator expected"}) {
            std::cerr << "FAIL: rejected atlas separator diagnostic message mismatch\n";
            return 51;
        }
    }

    {
        constexpr std::string_view source = " atlas shifted";
        scan_session session{};
        auto result = project_atlas_literal_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: atlas literal without brackets did not reject\n";
            return 52;
        }
        if (result.message != std::string{"atlas element group expected"}) {
            std::cerr << "FAIL: atlas literal without brackets returned wrong message: "
                      << result.message << "\n";
            return 53;
        }
        if (session.features.size() != 1 || !(session.features[0].kind == feature_kind::keyword)) {
            std::cerr << "FAIL: rejected atlas literal should only preserve the leading keyword feature\n";
            return 54;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected atlas literal should emit exactly one diagnostic\n";
            return 55;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected atlas literal diagnostic severity was not error\n";
            return 56;
        }
        if (session.diagnostics[0].message != std::string{"atlas element group expected"}) {
            std::cerr << "FAIL: rejected atlas literal diagnostic message mismatch\n";
            return 57;
        }
    }

    {
        constexpr std::string_view source = " manifold line = atlas[chart1, chart2]";
        scan_session session{};
        auto result = project_manifold_declaration_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: manifold declaration surface did not accept an atlas initializer\n";
            return 154;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: manifold declaration surface returned the wrong matched slice\n";
            return 155;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: manifold declaration surface did not consume the full declaration\n";
            return 156;
        }
        if (session.features.size() != 6) {
            std::cerr << "FAIL: expected manifold keyword + name + atlas initializer features, got "
                      << session.features.size() << "\n";
            return 157;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier &&
              session.features[2].kind == feature_kind::keyword &&
              session.features[3].kind == feature_kind::identifier &&
              session.features[4].kind == feature_kind::identifier &&
              session.features[5].kind == feature_kind::group)) {
            std::cerr << "FAIL: manifold declaration feature sequence did not preserve declaration -> atlas ordering\n";
            return 158;
        }
        if (session.features[1].semantic != std::string{"line"}) {
            std::cerr << "FAIL: manifold declaration name semantic label mismatch\n";
            return 159;
        }
        if (session.features[3].semantic != std::string{"chart1"}) {
            std::cerr << "FAIL: manifold declaration first atlas element semantic label mismatch\n";
            return 160;
        }
        if (session.features[4].semantic != std::string{"chart2"}) {
            std::cerr << "FAIL: manifold declaration second atlas element semantic label mismatch\n";
            return 161;
        }
        if (session.features[5].semantic != std::string{"atlas_elements"}) {
            std::cerr << "FAIL: manifold declaration atlas group semantic label mismatch\n";
            return 162;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete manifold declaration emitted unexpected diagnostics\n";
            return 163;
        }
    }

    {
        constexpr std::string_view source = " manifold line = ";
        scan_session session{};
        auto result = project_manifold_declaration_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete manifold declaration did not request more input\n";
            return 164;
        }
        if (result.message != std::string{"manifold initializer expected"}) {
            std::cerr << "FAIL: incomplete manifold declaration returned wrong message: "
                      << result.message << "\n";
            return 165;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: incomplete manifold declaration should preserve the keyword and name features\n";
            return 166;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete manifold declaration should emit exactly one diagnostic\n";
            return 167;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete manifold declaration diagnostic severity was not incomplete\n";
            return 168;
        }
        if (session.diagnostics[0].message != std::string{"manifold initializer expected"}) {
            std::cerr << "FAIL: incomplete manifold declaration diagnostic message mismatch\n";
            return 169;
        }
    }

    {
        constexpr std::string_view source = " manifold line atlas[chart1]";
        scan_session session{};
        auto result = project_manifold_declaration_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: manifold declaration without '=' did not reject\n";
            return 170;
        }
        if (result.message != std::string{"manifold '=' expected"}) {
            std::cerr << "FAIL: manifold declaration without '=' returned wrong message: "
                      << result.message << "\n";
            return 171;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: rejected manifold declaration should preserve the keyword and name features\n";
            return 172;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected manifold declaration should emit exactly one diagnostic\n";
            return 173;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected manifold declaration diagnostic severity was not error\n";
            return 174;
        }
        if (session.diagnostics[0].message != std::string{"manifold '=' expected"}) {
            std::cerr << "FAIL: rejected manifold declaration diagnostic message mismatch\n";
            return 175;
        }
    }

    {
        constexpr std::string_view source = " join_tag : int = 1;";
        scan_session session{};
        auto result = project_tag_declaration_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: tag declaration surface did not accept a bootstrap tag binding\n";
            return 235;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: tag declaration surface returned the wrong matched slice\n";
            return 236;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: tag declaration surface did not consume the full declaration\n";
            return 237;
        }
        if (session.features.size() != 4) {
            std::cerr << "FAIL: expected tag name + type + integer + declaration surface features, got "
                      << session.features.size() << "\n";
            return 238;
        }
        if (!(session.features[0].kind == feature_kind::identifier &&
              session.features[1].kind == feature_kind::keyword &&
              session.features[2].kind == feature_kind::integer &&
              session.features[3].kind == feature_kind::surface)) {
            std::cerr << "FAIL: tag declaration feature sequence did not preserve name -> type -> integer -> surface\n";
            return 239;
        }
        if (session.features[0].semantic != std::string{"join_tag"}) {
            std::cerr << "FAIL: tag declaration name semantic label mismatch\n";
            return 240;
        }
        if (session.features[1].semantic != std::string{"int"}) {
            std::cerr << "FAIL: tag declaration type semantic label mismatch\n";
            return 241;
        }
        if (session.features[2].semantic != std::string{"1"}) {
            std::cerr << "FAIL: tag declaration integer semantic label mismatch\n";
            return 242;
        }
        if (session.features[3].semantic != std::string{"bootstrap_tag_declaration"}) {
            std::cerr << "FAIL: tag declaration surface semantic label mismatch\n";
            return 243;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete tag declaration emitted unexpected diagnostics\n";
            return 244;
        }
    }

    {
        constexpr std::string_view source = " join_tag : int = ";
        scan_session session{};
        auto result = project_tag_declaration_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete tag declaration did not request more input\n";
            return 245;
        }
        if (result.message != std::string{"tag integer initializer expected"}) {
            std::cerr << "FAIL: incomplete tag declaration returned wrong message: "
                      << result.message << "\n";
            return 246;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: incomplete tag declaration should preserve the name and type features\n";
            return 247;
        }
        if (!(session.features[0].kind == feature_kind::identifier &&
              session.features[1].kind == feature_kind::keyword)) {
            std::cerr << "FAIL: incomplete tag declaration feature sequence was unstable\n";
            return 248;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete tag declaration should emit exactly one diagnostic\n";
            return 249;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete tag declaration diagnostic severity was not incomplete\n";
            return 250;
        }
        if (session.diagnostics[0].message != std::string{"tag integer initializer expected"}) {
            std::cerr << "FAIL: incomplete tag declaration diagnostic message mismatch\n";
            return 251;
        }
    }

    {
        constexpr std::string_view source = " coords[1.0, point - 10.0]";
        scan_session session{};
        auto parser = pure2_coords_literal();
        auto result = parser(source, 0, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: coords literal surface did not accept scalar + identifier tail elements\n";
            return 130;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: coords literal surface returned the wrong matched slice\n";
            return 131;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: coords literal surface did not consume the full literal\n";
            return 132;
        }
        if (session.features.size() != 5) {
            std::cerr << "FAIL: expected coords keyword + element heads/tail + group, got "
                      << session.features.size() << "\n";
            return 133;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::scalar &&
              session.features[2].kind == feature_kind::identifier &&
              session.features[3].kind == feature_kind::surface &&
              session.features[4].kind == feature_kind::group)) {
            std::cerr << "FAIL: coords literal feature sequence did not preserve element-level projection order\n";
            return 134;
        }
        if (session.features[1].semantic != std::string{"1.0"}) {
            std::cerr << "FAIL: coords literal scalar element semantic label mismatch\n";
            return 135;
        }
        if (session.features[2].semantic != std::string{"point"}) {
            std::cerr << "FAIL: coords literal identifier element semantic label mismatch\n";
            return 136;
        }
        if (session.features[3].semantic != std::string{"coords_element_tail"}) {
            std::cerr << "FAIL: coords literal tail semantic label mismatch\n";
            return 137;
        }
        if (session.features[4].semantic != std::string{"coords_elements"}) {
            std::cerr << "FAIL: coords literal group semantic label mismatch\n";
            return 138;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete coords literal emitted unexpected diagnostics\n";
            return 139;
        }
    }

    {
        constexpr std::string_view source = " coords[1.0, 2.";
        scan_session session{};
        auto parser = pure2_coords_literal();
        auto result = parser(source, 0, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete coords scalar element did not request more input\n";
            return 140;
        }
        if (result.message != std::string{"coords scalar fractional digits expected"}) {
            std::cerr << "FAIL: incomplete coords scalar element returned wrong message: "
                      << result.message << "\n";
            return 141;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: incomplete coords scalar element should preserve the keyword and completed first element\n";
            return 142;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete coords scalar element should emit exactly one diagnostic\n";
            return 143;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete coords scalar element diagnostic severity was not incomplete\n";
            return 144;
        }
        if (session.diagnostics[0].message != std::string{"coords scalar fractional digits expected"}) {
            std::cerr << "FAIL: incomplete coords scalar element diagnostic message mismatch\n";
            return 145;
        }
    }

    {
        constexpr std::string_view source = " local[0]";
        scan_session session{};
        auto parser = pure2_local_literal();
        auto result = parser(source, 0, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: local literal surface did not accept an indexed local element\n";
            return 146;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: local literal surface returned the wrong matched slice\n";
            return 147;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: local literal surface did not consume the full literal\n";
            return 148;
        }
        if (session.features.size() != 3) {
            std::cerr << "FAIL: expected local keyword + integer element + group, got "
                      << session.features.size() << "\n";
            return 149;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::integer &&
              session.features[2].kind == feature_kind::group)) {
            std::cerr << "FAIL: local literal feature sequence did not preserve integer element projection order\n";
            return 150;
        }
        if (session.features[1].semantic != std::string{"0"}) {
            std::cerr << "FAIL: local literal integer element semantic label mismatch\n";
            return 151;
        }
        if (session.features[2].semantic != std::string{"local_elements"}) {
            std::cerr << "FAIL: local literal group semantic label mismatch\n";
            return 152;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete local literal emitted unexpected diagnostics\n";
            return 153;
        }
    }

    {
        constexpr std::string_view source = " a j b";
        scan_session session{};
        auto result = project_join_expression_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: join surface did not accept a complete join expression\n";
            return 176;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: join surface returned the wrong matched slice\n";
            return 177;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: join surface did not consume the full expression\n";
            return 178;
        }
        if (session.features.size() != 3) {
            std::cerr << "FAIL: expected join lhs + operator + rhs features, got "
                      << session.features.size() << "\n";
            return 179;
        }
        if (!(session.features[0].kind == feature_kind::identifier &&
              session.features[1].kind == feature_kind::operator_token &&
              session.features[2].kind == feature_kind::identifier)) {
            std::cerr << "FAIL: join feature ordering was not lhs -> operator -> rhs\n";
            return 180;
        }
        if (session.features[0].semantic != std::string{"a"}) {
            std::cerr << "FAIL: join lhs semantic label mismatch\n";
            return 181;
        }
        if (session.features[1].semantic != std::string{"join"}) {
            std::cerr << "FAIL: join operator semantic label mismatch\n";
            return 182;
        }
        if (session.features[2].semantic != std::string{"b"}) {
            std::cerr << "FAIL: join rhs semantic label mismatch\n";
            return 183;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete join expression emitted unexpected diagnostics\n";
            return 184;
        }
    }

    {
        constexpr std::string_view source = " a j ";
        scan_session session{};
        auto result = project_join_expression_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete join expression did not request more input\n";
            return 185;
        }
        if (result.message != std::string{"join rhs expected"}) {
            std::cerr << "FAIL: incomplete join expression returned wrong message: "
                      << result.message << "\n";
            return 186;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: incomplete join expression should preserve lhs and operator features\n";
            return 187;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete join expression should emit exactly one diagnostic\n";
            return 188;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete join expression diagnostic severity was not incomplete\n";
            return 189;
        }
        if (session.diagnostics[0].message != std::string{"join rhs expected"}) {
            std::cerr << "FAIL: incomplete join expression diagnostic message mismatch\n";
            return 190;
        }
    }

    {
        constexpr std::string_view source = " a jb";
        scan_session session{};
        auto result = project_join_expression_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: malformed join operator placement did not reject\n";
            return 191;
        }
        if (result.message != std::string{"join operator expected"}) {
            std::cerr << "FAIL: malformed join operator placement returned wrong message: "
                      << result.message << "\n";
            return 192;
        }
        if (session.features.size() != 1 ||
            !(session.features[0].kind == feature_kind::identifier)) {
            std::cerr << "FAIL: malformed join operator placement should only preserve the lhs feature\n";
            return 193;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: malformed join operator placement should emit exactly one diagnostic\n";
            return 194;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: malformed join operator placement diagnostic severity was not error\n";
            return 195;
        }
        if (session.diagnostics[0].message != std::string{"join operator expected"}) {
            std::cerr << "FAIL: malformed join operator placement diagnostic message mismatch\n";
            return 196;
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
        if (session.features.size() != 17) {
            std::cerr << "FAIL: expected chart head + contains/project/embed body features, got "
                      << session.features.size() << "\n";
            return 52;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier &&
              session.features[2].kind == feature_kind::group &&
              session.features[3].kind == feature_kind::keyword &&
              session.features[4].kind == feature_kind::identifier &&
              session.features[5].kind == feature_kind::relation &&
              session.features[6].kind == feature_kind::scalar &&
              session.features[7].kind == feature_kind::keyword &&
              session.features[8].kind == feature_kind::keyword &&
              session.features[9].kind == feature_kind::identifier &&
              session.features[10].kind == feature_kind::group &&
              session.features[11].kind == feature_kind::keyword &&
              session.features[12].kind == feature_kind::group &&
              session.features[13].kind == feature_kind::keyword &&
              session.features[14].kind == feature_kind::integer &&
              session.features[15].kind == feature_kind::group &&
              session.features[16].kind == feature_kind::group)) {
            std::cerr << "FAIL: chart definition feature sequence did not preserve head + contains/project/embed body order\n";
            return 53;
        }
        if (session.features[2].semantic != std::string{"chart_parameters"}) {
            std::cerr << "FAIL: chart definition parameter group semantic label mismatch\n";
            return 54;
        }
        if (session.features[4].semantic != std::string{"point"}) {
            std::cerr << "FAIL: chart definition contains subject semantic label mismatch\n";
            return 55;
        }
        if (session.features[5].semantic != std::string{"<="}) {
            std::cerr << "FAIL: chart definition contains comparator semantic label mismatch\n";
            return 56;
        }
        if (session.features[6].semantic != std::string{"chart_contains_scalar_rhs"}) {
            std::cerr << "FAIL: chart definition contains scalar rhs semantic label mismatch\n";
            return 57;
        }
        if (session.features[8].semantic != std::string{"coords"}) {
            std::cerr << "FAIL: chart definition coords keyword semantic label mismatch\n";
            return 58;
        }
        if (session.features[9].semantic != std::string{"point"}) {
            std::cerr << "FAIL: chart definition coords identifier element semantic label mismatch\n";
            return 59;
        }
        if (session.features[10].semantic != std::string{"coords_elements"}) {
            std::cerr << "FAIL: chart definition coords element group semantic label mismatch\n";
            return 60;
        }
        if (session.features[12].semantic != std::string{"chart_embed_parameters"}) {
            std::cerr << "FAIL: chart definition embed parameter group semantic label mismatch\n";
            return 61;
        }
        if (session.features[13].semantic != std::string{"local"}) {
            std::cerr << "FAIL: chart definition embed local keyword semantic label mismatch\n";
            return 62;
        }
        if (session.features[14].semantic != std::string{"0"}) {
            std::cerr << "FAIL: chart definition embed local integer element semantic label mismatch\n";
            return 63;
        }
        if (session.features[15].semantic != std::string{"local_elements"}) {
            std::cerr << "FAIL: chart definition embed local element group semantic label mismatch\n";
            return 64;
        }
        if (session.features[16].semantic != std::string{"chart_body"}) {
            std::cerr << "FAIL: chart definition body group semantic label mismatch\n";
            return 65;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete chart definition emitted unexpected diagnostics\n";
            return 66;
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
        if (session.features.size() != 7) {
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
        if (session.features.size() != 16) {
            std::cerr << "FAIL: expected contains + project + embed clause features plus body group, got "
                      << session.features.size() << "\n";
            return 72;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier &&
              session.features[2].kind == feature_kind::relation &&
              session.features[3].kind == feature_kind::scalar &&
              session.features[4].kind == feature_kind::keyword &&
              session.features[5].kind == feature_kind::keyword &&
              session.features[6].kind == feature_kind::identifier &&
              session.features[7].kind == feature_kind::surface &&
              session.features[8].kind == feature_kind::group &&
              session.features[9].kind == feature_kind::keyword &&
              session.features[10].kind == feature_kind::group &&
              session.features[11].kind == feature_kind::keyword &&
              session.features[12].kind == feature_kind::integer &&
              session.features[13].kind == feature_kind::group &&
              session.features[14].kind == feature_kind::surface &&
              session.features[15].kind == feature_kind::group)) {
            std::cerr << "FAIL: chart body feature sequence did not preserve clause-first projection order\n";
            return 73;
        }
        if (session.features[1].semantic != std::string{"point"}) {
            std::cerr << "FAIL: chart contains subject semantic label mismatch\n";
            return 74;
        }
        if (session.features[2].semantic != std::string{"<="}) {
            std::cerr << "FAIL: chart contains comparator semantic label mismatch\n";
            return 75;
        }
        if (session.features[3].semantic != std::string{"chart_contains_scalar_rhs"}) {
            std::cerr << "FAIL: chart contains scalar rhs semantic label mismatch\n";
            return 76;
        }
        if (session.features[5].semantic != std::string{"coords"}) {
            std::cerr << "FAIL: chart project coords keyword semantic label mismatch\n";
            return 77;
        }
        if (session.features[6].semantic != std::string{"point"}) {
            std::cerr << "FAIL: chart project coords identifier element semantic label mismatch\n";
            return 78;
        }
        if (session.features[7].semantic != std::string{"coords_element_tail"}) {
            std::cerr << "FAIL: chart project coords element tail semantic label mismatch\n";
            return 79;
        }
        if (session.features[8].semantic != std::string{"coords_elements"}) {
            std::cerr << "FAIL: chart project coords element group semantic label mismatch\n";
            return 80;
        }
        if (session.features[10].semantic != std::string{"chart_embed_parameters"}) {
            std::cerr << "FAIL: chart embed parameter group semantic label mismatch\n";
            return 81;
        }
        if (session.features[11].semantic != std::string{"local"}) {
            std::cerr << "FAIL: chart embed local keyword semantic label mismatch\n";
            return 82;
        }
        if (session.features[12].semantic != std::string{"0"}) {
            std::cerr << "FAIL: chart embed local integer element semantic label mismatch\n";
            return 83;
        }
        if (session.features[13].semantic != std::string{"local_elements"}) {
            std::cerr << "FAIL: chart embed local element group semantic label mismatch\n";
            return 84;
        }
        if (session.features[14].semantic != std::string{"chart_embed_expression_tail"}) {
            std::cerr << "FAIL: chart embed expression tail semantic label mismatch\n";
            return 85;
        }
        if (session.features[15].semantic != std::string{"chart_body"}) {
            std::cerr << "FAIL: chart body group semantic label mismatch\n";
            return 86;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete chart body emitted unexpected diagnostics\n";
            return 87;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  embed(local) -> a j b\n"
            "}";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: chart body did not accept join-specific embed expression\n";
            return 197;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: chart body join embed expression returned the wrong matched slice\n";
            return 198;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: chart body join embed expression did not consume the full body\n";
            return 199;
        }
        if (session.features.size() != 6) {
            std::cerr << "FAIL: expected embed head + join lhs/operator/rhs + body group, got "
                      << session.features.size() << "\n";
            return 200;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::group &&
              session.features[2].kind == feature_kind::identifier &&
              session.features[3].kind == feature_kind::operator_token &&
              session.features[4].kind == feature_kind::identifier &&
              session.features[5].kind == feature_kind::group)) {
            std::cerr << "FAIL: chart body join embed feature ordering was not embed -> bindings -> lhs -> operator -> rhs -> body\n";
            return 201;
        }
        if (session.features[1].semantic != std::string{"chart_embed_parameters"}) {
            std::cerr << "FAIL: chart body join embed binding group semantic label mismatch\n";
            return 202;
        }
        if (session.features[2].semantic != std::string{"a"}) {
            std::cerr << "FAIL: chart body join embed lhs semantic label mismatch\n";
            return 203;
        }
        if (session.features[3].semantic != std::string{"join"}) {
            std::cerr << "FAIL: chart body join embed operator semantic label mismatch\n";
            return 204;
        }
        if (session.features[4].semantic != std::string{"b"}) {
            std::cerr << "FAIL: chart body join embed rhs semantic label mismatch\n";
            return 205;
        }
        if (session.features[5].semantic != std::string{"chart_body"}) {
            std::cerr << "FAIL: chart body join embed group semantic label mismatch\n";
            return 206;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: chart body join embed expression emitted unexpected diagnostics\n";
            return 207;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  contains point 20.0\n"
            "}";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: chart body without a contains comparator did not reject\n";
            return 85;
        }
        if (result.message != std::string{"contains comparator expected"}) {
            std::cerr << "FAIL: chart body without a contains comparator returned wrong message: "
                      << result.message << "\n";
            return 86;
        }
        if (session.features.size() != 2) {
            std::cerr << "FAIL: rejected contains clause should preserve the contains keyword and subject\n";
            return 87;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::identifier)) {
            std::cerr << "FAIL: rejected contains clause did not preserve keyword -> identifier ordering\n";
            return 88;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected contains clause should emit exactly one diagnostic\n";
            return 89;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected contains clause diagnostic severity was not error\n";
            return 90;
        }
        if (session.diagnostics[0].message != std::string{"contains comparator expected"}) {
            std::cerr << "FAIL: rejected contains clause diagnostic message mismatch\n";
            return 91;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  contains point <= ";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete contains rhs did not request more input\n";
            return 92;
        }
        if (result.message != std::string{"contains rhs expected"}) {
            std::cerr << "FAIL: incomplete contains rhs returned wrong message: "
                      << result.message << "\n";
            return 93;
        }
        if (session.features.size() != 3) {
            std::cerr << "FAIL: incomplete contains rhs should preserve keyword, subject, and comparator\n";
            return 94;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete contains rhs should emit exactly one diagnostic\n";
            return 95;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete contains rhs diagnostic severity was not incomplete\n";
            return 96;
        }
        if (session.diagnostics[0].message != std::string{"contains rhs expected"}) {
            std::cerr << "FAIL: incomplete contains rhs diagnostic message mismatch\n";
            return 97;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  contains point <= 20.";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete contains scalar literal did not request more input\n";
            return 122;
        }
        if (result.message != std::string{"contains scalar fractional digits expected"}) {
            std::cerr << "FAIL: incomplete contains scalar literal returned wrong message: "
                      << result.message << "\n";
            return 123;
        }
        if (session.features.size() != 3) {
            std::cerr << "FAIL: incomplete contains scalar literal should preserve keyword, subject, and comparator\n";
            return 124;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete contains scalar literal should emit exactly one diagnostic\n";
            return 125;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete contains scalar literal diagnostic severity was not incomplete\n";
            return 126;
        }
        if (session.diagnostics[0].message != std::string{"contains scalar fractional digits expected"}) {
            std::cerr << "FAIL: incomplete contains scalar literal diagnostic message mismatch\n";
            return 127;
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
            return 98;
        }
        if (result.message != std::string{"coords element group incomplete"}) {
            std::cerr << "FAIL: incomplete coords literal returned wrong message: "
                      << result.message << "\n";
            return 99;
        }
        if (session.features.size() != 3) {
            std::cerr << "FAIL: incomplete coords literal should preserve project + coords keyword and completed element features\n";
            return 100;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete coords literal should emit exactly one diagnostic\n";
            return 101;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete coords literal diagnostic severity was not incomplete\n";
            return 102;
        }
        if (session.diagnostics[0].message != std::string{"coords element group incomplete"}) {
            std::cerr << "FAIL: incomplete coords literal diagnostic message mismatch\n";
            return 103;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  project -> coords[point]\n"
            "  embed(local) -> local[\n"
            "}";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: incomplete local literal did not request more input\n";
            return 104;
        }
        if (result.message != std::string{"local element group incomplete"}) {
            std::cerr << "FAIL: incomplete local literal returned wrong message: "
                      << result.message << "\n";
            return 105;
        }
        if (session.features.size() != 7) {
            std::cerr << "FAIL: incomplete local literal should preserve project clause, embed head, and local keyword features\n";
            return 106;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete local literal should emit exactly one diagnostic\n";
            return 107;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete local literal diagnostic severity was not incomplete\n";
            return 108;
        }
        if (session.diagnostics[0].message != std::string{"local element group incomplete"}) {
            std::cerr << "FAIL: incomplete local literal diagnostic message mismatch\n";
            return 109;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  embed(local) -> line.transition(\"identity\", \"shifted\", coords[17.0])\n"
            "}";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: chart body did not accept transition feature-stream expression\n";
            return 122;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: chart body transition expression returned the wrong matched slice\n";
            return 123;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: chart body transition expression did not consume the full body\n";
            return 124;
        }
        if (session.features.size() != 11) {
            std::cerr << "FAIL: expected embed head + transition receiver/keyword/args + nested coords + trailing groups, got "
                      << session.features.size() << "\n";
            return 125;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[1].kind == feature_kind::group &&
              session.features[2].kind == feature_kind::identifier &&
              session.features[3].kind == feature_kind::keyword &&
              session.features[4].kind == feature_kind::surface &&
              session.features[5].kind == feature_kind::surface &&
              session.features[6].kind == feature_kind::keyword &&
              session.features[7].kind == feature_kind::scalar &&
              session.features[8].kind == feature_kind::group &&
              session.features[9].kind == feature_kind::group &&
              session.features[10].kind == feature_kind::group)) {
            std::cerr << "FAIL: transition feature ordering was not embed -> bindings -> receiver -> transition -> strings -> nested coords -> argument/body groups\n";
            return 126;
        }
        if (session.features[2].semantic != std::string{"line"}) {
            std::cerr << "FAIL: transition receiver semantic label mismatch\n";
            return 127;
        }
        if (session.features[3].semantic != std::string{"transition"}) {
            std::cerr << "FAIL: transition keyword semantic label mismatch\n";
            return 128;
        }
        if (session.features[6].semantic != std::string{"coords"}) {
            std::cerr << "FAIL: transition nested coords keyword semantic label mismatch\n";
            return 129;
        }
        if (session.features[7].semantic != std::string{"17.0"}) {
            std::cerr << "FAIL: transition nested coords scalar semantic label mismatch\n";
            return 130;
        }
        if (session.features[8].semantic != std::string{"coords_elements"}) {
            std::cerr << "FAIL: transition nested coords group semantic label mismatch\n";
            return 131;
        }
        if (session.features[9].semantic != std::string{"transition_arguments"}) {
            std::cerr << "FAIL: transition argument group semantic label mismatch\n";
            return 132;
        }
        if (session.features[10].semantic != std::string{"chart_body"}) {
            std::cerr << "FAIL: chart body group semantic label mismatch after transition expression\n";
            return 133;
        }
        if (session.features[4].semantic != std::string{"identity"} ||
            session.features[5].semantic != std::string{"shifted"}) {
            std::cerr << "FAIL: transition string argument semantic labels mismatch\n";
            return 134;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete transition expression emitted unexpected diagnostics\n";
            return 135;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  embed(local) -> line.transition(\"identity\" \"shifted\", coords[17.0])\n"
            "}";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: transition expression without first comma did not reject\n";
            return 135;
        }
        if (result.message != std::string{"transition argument separator expected"}) {
            std::cerr << "FAIL: transition expression without first comma returned wrong message: "
                      << result.message << "\n";
            return 136;
        }
        if (session.diagnostics.size() != 1 ||
            !(session.diagnostics[0].severity == diagnostic_severity::error) ||
            session.diagnostics[0].message != std::string{"transition argument separator expected"}) {
            std::cerr << "FAIL: transition missing comma diagnostic was unstable\n";
            return 137;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  embed(local) -> line.transition(\"identity\", \"shifted\", coords[17.0]\n"
            "}";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: transition expression without closing ')' did not reject\n";
            return 138;
        }
        if (result.message != std::string{"transition argument list incomplete"}) {
            std::cerr << "FAIL: transition expression without closing ')' returned wrong message: "
                      << result.message << "\n";
            return 139;
        }
        if (session.diagnostics.size() != 1 ||
            !(session.diagnostics[0].severity == diagnostic_severity::error) ||
            session.diagnostics[0].message != std::string{"transition argument list incomplete"}) {
            std::cerr << "FAIL: transition missing ')' diagnostic was unstable\n";
            return 140;
        }
    }

    {
        constexpr std::string_view source =
            "{\n"
            "  embed(local) -> line.transition(\"identity\", \"shifted\", coords[17.";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: transition expression with incomplete coords payload did not request more input\n";
            return 141;
        }
        if (result.message != std::string{"coords scalar fractional digits expected"}) {
            std::cerr << "FAIL: transition expression with incomplete coords payload returned wrong message: "
                      << result.message << "\n";
            return 142;
        }
        if (session.diagnostics.size() != 1 ||
            !(session.diagnostics[0].severity == diagnostic_severity::incomplete) ||
            session.diagnostics[0].message != std::string{"coords scalar fractional digits expected"}) {
            std::cerr << "FAIL: transition incomplete coords diagnostic was unstable\n";
            return 143;
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
            return 110;
        }
        if (result.message != std::string{"embed expression expected"}) {
            std::cerr << "FAIL: incomplete embed clause returned wrong message: "
                      << result.message << "\n";
            return 111;
        }
        if (session.features.size() != 6) {
            std::cerr << "FAIL: incomplete chart body should preserve project clause and embed head features\n";
            return 112;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: incomplete chart body should emit exactly one diagnostic\n";
            return 113;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::incomplete)) {
            std::cerr << "FAIL: incomplete chart body diagnostic severity was not incomplete\n";
            return 114;
        }
        if (session.diagnostics[0].message != std::string{"embed expression expected"}) {
            std::cerr << "FAIL: incomplete chart body diagnostic message mismatch\n";
            return 115;
        }
    }

    {
        constexpr std::string_view source = "{ project coords[point] }";
        scan_session session{};
        auto result = project_chart_body_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: chart body without clause arrow did not reject\n";
            return 116;
        }
        if (result.message != std::string{"project arrow expected"}) {
            std::cerr << "FAIL: chart body without clause arrow returned wrong message: "
                      << result.message << "\n";
            return 117;
        }
        if (session.features.size() != 1 || !(session.features[0].kind == feature_kind::keyword)) {
            std::cerr << "FAIL: rejected chart body should only preserve the project keyword feature\n";
            return 118;
        }
        if (session.diagnostics.size() != 1) {
            std::cerr << "FAIL: rejected chart body should emit exactly one diagnostic\n";
            return 119;
        }
        if (!(session.diagnostics[0].severity == diagnostic_severity::error)) {
            std::cerr << "FAIL: rejected chart body diagnostic severity was not error\n";
            return 120;
        }
        if (session.diagnostics[0].message != std::string{"project arrow expected"}) {
            std::cerr << "FAIL: rejected chart body diagnostic message mismatch\n";
            return 121;
        }
    }

    {
        constexpr std::string_view source =
            "\n"
            "chart identity(point: f64) {\n"
            "  contains point <= 20.0\n"
            "  project -> coords[point]\n"
            "  embed(local) -> local[0]\n"
            "}\n"
            "manifold line = atlas[shifted, identity]\n"
            "coords[1.0, point - 10.0]\n"
            "a j b\n"
            "line.transition(\"identity\", \"shifted\", coords[17.0])\n";
        scan_session session{};
        auto result = project_translation_unit_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: translation unit surface did not accept sequential pure2 surfaces\n";
            return 208;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: translation unit surface returned the wrong matched slice\n";
            return 209;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: translation unit surface did not consume the full source\n";
            return 210;
        }
        if (session.features.size() != 40) {
            std::cerr << "FAIL: expected chart + manifold + coords + join + transition + translation unit features, got "
                      << session.features.size() << "\n";
            return 211;
        }
        if (!(session.features[0].kind == feature_kind::keyword &&
              session.features[16].kind == feature_kind::group &&
              session.features[17].kind == feature_kind::keyword &&
              session.features[22].kind == feature_kind::group &&
              session.features[23].kind == feature_kind::keyword &&
              session.features[27].kind == feature_kind::group &&
              session.features[28].kind == feature_kind::identifier &&
              session.features[29].kind == feature_kind::operator_token &&
              session.features[30].kind == feature_kind::identifier &&
              session.features[31].kind == feature_kind::identifier &&
              session.features[38].kind == feature_kind::group &&
              session.features[39].kind == feature_kind::group)) {
            std::cerr << "FAIL: translation unit feature boundaries did not preserve top-level surface order\n";
            return 212;
        }
        if (session.features[2].semantic != std::string{"chart_parameters"}) {
            std::cerr << "FAIL: translation unit chart parameter group semantic label mismatch\n";
            return 213;
        }
        if (session.features[16].semantic != std::string{"chart_body"}) {
            std::cerr << "FAIL: translation unit chart body group semantic label mismatch\n";
            return 214;
        }
        if (session.features[18].semantic != std::string{"line"}) {
            std::cerr << "FAIL: translation unit manifold name semantic label mismatch\n";
            return 215;
        }
        if (session.features[22].semantic != std::string{"atlas_elements"}) {
            std::cerr << "FAIL: translation unit atlas group semantic label mismatch\n";
            return 216;
        }
        if (session.features[24].semantic != std::string{"1.0"}) {
            std::cerr << "FAIL: translation unit coords scalar semantic label mismatch\n";
            return 217;
        }
        if (session.features[26].semantic != std::string{"coords_element_tail"}) {
            std::cerr << "FAIL: translation unit coords tail semantic label mismatch\n";
            return 218;
        }
        if (session.features[29].semantic != std::string{"join"}) {
            std::cerr << "FAIL: translation unit join operator semantic label mismatch\n";
            return 219;
        }
        if (session.features[31].semantic != std::string{"line"}) {
            std::cerr << "FAIL: translation unit transition receiver semantic label mismatch\n";
            return 220;
        }
        if (session.features[32].semantic != std::string{"transition"}) {
            std::cerr << "FAIL: translation unit transition keyword semantic label mismatch\n";
            return 221;
        }
        if (session.features[35].semantic != std::string{"coords"}) {
            std::cerr << "FAIL: translation unit nested coords keyword semantic label mismatch\n";
            return 222;
        }
        if (session.features[36].semantic != std::string{"17.0"}) {
            std::cerr << "FAIL: translation unit nested coords scalar semantic label mismatch\n";
            return 223;
        }
        if (session.features[38].semantic != std::string{"transition_arguments"}) {
            std::cerr << "FAIL: translation unit transition argument group semantic label mismatch\n";
            return 224;
        }
        if (session.features[39].semantic != std::string{"translation_unit"}) {
            std::cerr << "FAIL: translation unit outer group semantic label mismatch\n";
            return 225;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: complete translation unit emitted unexpected diagnostics\n";
            return 226;
        }
    }

    {
        const std::string source_path =
            std::string{CPPFORT_SOURCE_DIR} + "/src/selfhost/bootstrap_tags.cpp2";
        std::ifstream input(source_path);
        if (!input) {
            std::cerr << "FAIL: could not open bootstrap tag source at " << source_path << "\n";
            return 252;
        }

        std::ostringstream buffer;
        buffer << input.rdbuf();
        const std::string source = buffer.str();

        scan_session session{};
        auto result = project_translation_unit_feature_stream(source, session);

        if (!(result.outcome == scan_signal::accept)) {
            std::cerr << "FAIL: translation unit did not accept bootstrap_tags.cpp2\n";
            return 253;
        }
        if (!result.value.has_value() || result.value.value() != source) {
            std::cerr << "FAIL: bootstrap tag translation unit returned the wrong matched slice\n";
            return 254;
        }
        if (result.consumed != static_cast<int>(source.size())) {
            std::cerr << "FAIL: bootstrap tag translation unit did not consume the full source\n";
            return 255;
        }
        if (session.features.size() != 93) {
            std::cerr << "FAIL: expected 23 tag declarations plus translation unit feature, got "
                      << session.features.size() << "\n";
            return 256;
        }

        int declaration_count = 0;
        for (const auto& feature : session.features) {
            if (feature.kind == feature_kind::surface &&
                feature.semantic == std::string{"bootstrap_tag_declaration"}) {
                declaration_count += 1;
            }
        }
        if (declaration_count != 23) {
            std::cerr << "FAIL: expected 23 bootstrap tag declaration surfaces, got "
                      << declaration_count << "\n";
            return 257;
        }
        if (session.features[0].semantic != std::string{"join_tag"} ||
            session.features[1].semantic != std::string{"int"} ||
            session.features[2].semantic != std::string{"1"} ||
            session.features[3].semantic != std::string{"bootstrap_tag_declaration"}) {
            std::cerr << "FAIL: bootstrap tag translation unit did not preserve first declaration features\n";
            return 258;
        }
        if (session.features[92].semantic != std::string{"translation_unit"}) {
            std::cerr << "FAIL: bootstrap tag translation unit outer group semantic label mismatch\n";
            return 259;
        }
        if (!session.diagnostics.empty()) {
            std::cerr << "FAIL: bootstrap tag translation unit emitted unexpected diagnostics\n";
            return 260;
        }
    }

    {
        constexpr std::string_view source = "local[0]";
        scan_session session{};
        auto result = project_translation_unit_feature_stream(source, session);

        if (!(result.outcome == scan_signal::reject)) {
            std::cerr << "FAIL: translation unit accepted an unsupported top-level local literal\n";
            return 227;
        }
        if (result.message != std::string{"top-level surface expected"}) {
            std::cerr << "FAIL: translation unit unsupported top-level message mismatch: "
                      << result.message << "\n";
            return 228;
        }
        if (!session.features.empty()) {
            std::cerr << "FAIL: unsupported top-level translation unit should not preserve features\n";
            return 229;
        }
        if (session.diagnostics.size() != 1 ||
            !(session.diagnostics[0].severity == diagnostic_severity::error) ||
            session.diagnostics[0].message != std::string{"top-level surface expected"}) {
            std::cerr << "FAIL: translation unit unsupported top-level diagnostic was unstable\n";
            return 230;
        }
    }

    {
        constexpr std::string_view source =
            "chart identity(point: f64) {\n"
            "  contains point <= 20.0\n"
            "  project -> coords[point]\n"
            "  embed(local) -> local[0]\n"
            "}\n"
            "manifold line = ";
        scan_session session{};
        auto result = project_translation_unit_feature_stream(source, session);

        if (!(result.outcome == scan_signal::need_more)) {
            std::cerr << "FAIL: translation unit did not request more input for a truncated trailing manifold\n";
            return 231;
        }
        if (result.message != std::string{"manifold initializer expected"}) {
            std::cerr << "FAIL: translation unit truncated manifold message mismatch: "
                      << result.message << "\n";
            return 232;
        }
        if (session.features.size() != 19) {
            std::cerr << "FAIL: translation unit should preserve the full chart definition plus manifold head before stalling\n";
            return 233;
        }
        if (session.diagnostics.size() != 1 ||
            !(session.diagnostics[0].severity == diagnostic_severity::incomplete) ||
            session.diagnostics[0].message != std::string{"manifold initializer expected"}) {
            std::cerr << "FAIL: translation unit truncated manifold diagnostic was unstable\n";
            return 234;
        }
    }

    std::cout << "selfhost rbcursive smoke passed\n";
    return 0;
}
