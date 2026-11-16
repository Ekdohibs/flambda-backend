(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)


(* The meets of the heads of values commute with a renaming of
   variables. Lemmas only: nothing here is a statement to review.

   For [meet_variant], [meet_head_of_kind_value_non_null] and
   [meet_head_of_kind_value] of FlambdaMeet.v, a lemma named after the
   function says that on the renamed environment and renamed arguments
   it gives the renamed answer. For the two meets of heads the lemma
   is [meet_of_parts_related] of the function
   (FlambdaMeetRenameSpec.v). There are two joins and two meets of
   less fuel, one with the arguments and one with the renamed
   arguments, which the renaming relates, and the meets of the two
   row-like structures with each, which it relates too. Those are
   hypotheses of the section; the owner of the meet proves them of its
   meets. The meet of two alternatives is
   FlambdaMeetRenameScopesProofs.v, the meets of the parts that are
   not values FlambdaEnvRenameMeetProductsProofs.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningBoolSpec FlambdaTypeEqualSpec FlambdaTypeRenameSpec
  FlambdaTypeRenameProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaProversRenameSpec FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetRenameSpec FlambdaEnvRenameMeetProofs
  FlambdaEnvRenameMeetPartsProofs FlambdaEnvRenameMeetProductsProofs
  FlambdaMeetRenameScopesProofs.
Import ListNotations.

(* What [meet_variant] answers, renamed: the two variables, the
   blocks, the immediates and the extensions. *)
Definition rename_variant_parts (r : renaming)
    (parts :
      option variable * option variable * Or_unknown.t row_like_for_blocks
      * Or_unknown.t flambda_type * variant_extensions)
    : option variable * option variable * Or_unknown.t row_like_for_blocks
      * Or_unknown.t flambda_type * variant_extensions :=
  let '(is_int, get_tag, blocks, imms, extensions) := parts in
  (option_map (rename_variable r) is_int,
   option_map (rename_variable r) get_tag,
   or_unknown_map (rename_row_like_for_blocks r) blocks,
   or_unknown_map (rename_type r) imms,
   rename_variant_extensions r extensions).

Section Values.

Variable r : renaming.
Variables join join' : join_scoped_envs.
Variable choices : meet_choices.
Variables m m' : meet_expanded_head.

Hypothesis Hjoin : join_related_at r join join' m m'.
Hypothesis Hm : meet_expanded_head_related r m m'.

(* The renaming relates the meets of the two row-like structures. *)
Hypothesis Hblocks :
  meet_of_parts_related r (rename_row_like_for_blocks r)
    (rename_row_like_for_blocks r) (meet_row_like_for_blocks join choices m)
    (meet_row_like_for_blocks join' choices m').
Hypothesis Hclosures :
  meet_of_parts_related r (rename_row_like_for_closures r)
    (rename_row_like_for_closures r)
    (meet_row_like_for_closures join choices m)
    (meet_row_like_for_closures join' choices m').

(* ======================================================================== *)
(* Variants                                                                 *)
(* ======================================================================== *)

(* The two alternatives of a variant: its immediates, and its blocks
   with the variable for their tag. *)

Lemma meet_immediates_renamed :
  meet_of_parts_related r (or_unknown_map (rename_type r))
    (or_unknown_map (rename_type r))
    (meet_unknown (meet m) is_obviously_bottom)
    (meet_unknown (meet m') is_obviously_bottom).
Proof.
  intros env imms1 imms2.
  apply meet_unknown_renamed;
    [ exact (meet_of_types_renamed r m m' Hm)
    | exact (is_obviously_bottom_renamed r) ].
Qed.

Lemma meet_tag_and_blocks_renamed :
  meet_of_parts_related r
    (fun x : option variable * Or_unknown.t row_like_for_blocks =>
       (option_map (rename_variable r) (fst x),
        or_unknown_map (rename_row_like_for_blocks r) (snd x)))
    (fun x : option variable * Or_unknown.t row_like_for_blocks =>
       (option_map (rename_variable r) (fst x),
        or_unknown_map (rename_row_like_for_blocks r) (snd x)))
    (fun env left right =>
       combine_results2 env (meet_relation m)
         (meet_unknown (meet_row_like_for_blocks join choices m)
            row_like_for_blocks_is_bottom)
         (fst left) (fst right) (snd left) (snd right)
         (fun get_tag blocks => (get_tag, blocks)))
    (fun env left right =>
       combine_results2 env (meet_relation m')
         (meet_unknown (meet_row_like_for_blocks join' choices m')
            row_like_for_blocks_is_bottom)
         (fst left) (fst right) (snd left) (snd right)
         (fun get_tag blocks => (get_tag, blocks))).
Proof.
  intros env [get_tag1 blocks1] [get_tag2 blocks2]. cbn [fst snd].
  apply
    (combine_results2_renamed r (option_map (rename_variable r))
       (or_unknown_map (rename_row_like_for_blocks r))
       (fun x : option variable * Or_unknown.t row_like_for_blocks =>
          (option_map (rename_variable r) (fst x),
           or_unknown_map (rename_row_like_for_blocks r) (snd x)))).
  - exact (meet_relation_renamed r m m' Hm).
  - intros env' blocks blocks'.
    apply meet_unknown_renamed;
      [ exact Hblocks | exact (row_like_for_blocks_is_bottom_renamed r) ].
  - reflexivity.
Qed.

Lemma meet_variant_renamed (env : Meet_env.t)
    (is_int1 get_tag1 : option variable)
    (blocks1 : Or_unknown.t row_like_for_blocks)
    (imms1 : Or_unknown.t flambda_type) (is_int2 get_tag2 : option variable)
    (blocks2 : Or_unknown.t row_like_for_blocks)
    (imms2 : Or_unknown.t flambda_type)
    (extensions1 extensions2 : variant_extensions) :
  meet_variant join' choices m' (rename_meet_env r env)
    (option_map (rename_variable r) is_int1)
    (option_map (rename_variable r) get_tag1)
    (or_unknown_map (rename_row_like_for_blocks r) blocks1)
    (or_unknown_map (rename_type r) imms1)
    (option_map (rename_variable r) is_int2)
    (option_map (rename_variable r) get_tag2)
    (or_unknown_map (rename_row_like_for_blocks r) blocks2)
    (or_unknown_map (rename_type r) imms2)
    (rename_variant_extensions r extensions1)
    (rename_variant_extensions r extensions2)
  = option_map (rename_meet_result r (rename_variant_parts r))
      (meet_variant join choices m env is_int1 get_tag1 blocks1 imms1 is_int2
         get_tag2 blocks2 imms2 extensions1 extensions2).
Proof.
  unfold meet_variant. cbv zeta.
  apply
    (combine_results2_renamed r (option_map (rename_variable r))
       (fun rest :
              option variable * Or_unknown.t row_like_for_blocks
              * Or_unknown.t flambda_type * variant_extensions =>
          (option_map (rename_variable r) (fst (fst (fst rest))),
           or_unknown_map (rename_row_like_for_blocks r)
             (snd (fst (fst rest))),
           or_unknown_map (rename_type r) (snd (fst rest)),
           rename_variant_extensions r (snd rest)))
       (rename_variant_parts r) env _ _ _ _ is_int1 is_int2
       (get_tag1, blocks1, imms1, extensions1)
       (get_tag2, blocks2, imms2, extensions2)).
  - exact (meet_relation_renamed r m m' Hm).
  - intros env' [[[get_tag blocks] imms] extensions]
      [[[get_tag' blocks'] imms'] extensions'].
    cbn [fst snd].
    rewrite
      (meet_disjunction_renamed r join join' choices m m' Hjoin Hm
         (fun r => or_unknown_map (rename_type r))
         (fun r (x : option variable * Or_unknown.t row_like_for_blocks) =>
            (option_map (rename_variable r) (fst x),
             or_unknown_map (rename_row_like_for_blocks r) (snd x)))
         _ _ _ _ _ _ meet_immediates_renamed meet_tag_and_blocks_renamed
         eq_refl eq_refl env' imms (get_tag, blocks) extensions imms'
         (get_tag', blocks') extensions').
    match goal with
    | |- option_map _ (option_map _ ?call) = _ =>
      destruct call as [result | ]
    end;
      cbn [option_map]; [ | reflexivity ].
    f_equal. apply map_result_renamed.
    intros [[imms'' [get_tag'' blocks'']] extensions'']. reflexivity.
  - intros is_int [[[get_tag blocks] imms] extensions]. reflexivity.
Qed.

(* ======================================================================== *)
(* The heads of values                                                      *)
(* ======================================================================== *)

Lemma meet_head_of_kind_value_non_null_renamed :
  meet_of_parts_related r (rename_head_of_kind_value_non_null r)
    (rename_head_of_kind_value_non_null r)
    (meet_head_of_kind_value_non_null join choices m)
    (meet_head_of_kind_value_non_null join' choices m').
Proof.
  intros env head1 head2.
  destruct head1
    as [is_int1 imms1 get_tag1 blocks1 extensions1 is_unique1 | alloc_mode1
       | k1 n1 alloc_mode1 | by_function_slot1 alloc_mode1 | strs1],
    head2
    as [is_int2 imms2 get_tag2 blocks2 extensions2 is_unique2 | alloc_mode2
       | k2 n2 alloc_mode2 | by_function_slot2 alloc_mode2 | strs2];
    cbn [rename_head_of_kind_value_non_null meet_head_of_kind_value_non_null];
    try reflexivity.
  - (* Two variants. *)
    cbv zeta.
    change
      (match extensions1 with
       | No_extensions => No_extensions
       | Ext when_immediate when_block =>
         Ext (rename_env_extension r when_immediate)
           (rename_env_extension r when_block)
       end)
      with (rename_variant_extensions r extensions1).
    change
      (match extensions2 with
       | No_extensions => No_extensions
       | Ext when_immediate when_block =>
         Ext (rename_env_extension r when_immediate)
           (rename_env_extension r when_block)
       end)
      with (rename_variant_extensions r extensions2).
    rewrite meet_variant_renamed.
    destruct
      (meet_variant join choices m env is_int1 get_tag1 blocks1 imms1 is_int2
         get_tag2 blocks2 imms2 extensions1 extensions2)
      as [result | ];
      cbn [option_map]; [ | reflexivity ].
    f_equal. apply map_result_renamed.
    intros [[[[is_int get_tag] blocks] imms] extensions]. reflexivity.
  - (* A variant and a mutable block. *)
    destruct blocks1 as [[known1 other1 alloc_mode1] | ]; [ | reflexivity ].
    destruct alloc_mode1, alloc_mode2; reflexivity.
  - (* A mutable block and a variant. *)
    destruct blocks2 as [[known2 other2 alloc_mode2] | ]; [ | reflexivity ].
    destruct alloc_mode1, alloc_mode2; reflexivity.
  - (* Two mutable blocks. *)
    destruct alloc_mode1, alloc_mode2; reflexivity.
  - (* Two boxed numbers. *)
    destruct (boxable_number_eqb k1 k2); [ | reflexivity ].
    apply
      (combine_results2_renamed r (rename_type r)
         (fun alloc_mode : Alloc_mode.t => alloc_mode)
         (rename_head_of_kind_value_non_null r)).
    + exact (meet_of_types_renamed r m m' Hm).
    + intros env' mode mode'. cbn [option_map]. f_equal.
      apply meet_alloc_mode_renamed.
    + reflexivity.
  - (* Two sets of closures. *)
    apply
      (combine_results2_renamed r (rename_row_like_for_closures r)
         (fun alloc_mode : Alloc_mode.t => alloc_mode)
         (rename_head_of_kind_value_non_null r)).
    + exact Hclosures.
    + intros env' mode mode'. cbn [option_map]. f_equal.
      apply meet_alloc_mode_renamed.
    + reflexivity.
  - (* Two strings. *)
    rewrite
      (set_meet_renamed r (lists_eqb Z.eqb) (fun s => s) (fun s => s) env
         strs1 strs2 (fun _ => eq_refl)).
    destruct (set_meet (lists_eqb Z.eqb) (fun s => s) env strs1 strs2)
      as [res | [ | | | strs] env'];
      reflexivity.
Qed.

(* The two alternatives of a value: it is not null, or it is null. *)

Lemma meet_non_null_renamed :
  meet_of_parts_related r
    (or_unknown_or_bottom_map (rename_head_of_kind_value_non_null r))
    (or_unknown_or_bottom_map (rename_head_of_kind_value_non_null r))
    (meet_or_unknown_or_bottom
       (fun env elt1 elt2 =>
          option_map
            (map_result
               (fun x : head_of_kind_value_non_null =>
                  @Ok head_of_kind_value_non_null x))
            (meet_head_of_kind_value_non_null join choices m env elt1 elt2)))
    (meet_or_unknown_or_bottom
       (fun env elt1 elt2 =>
          option_map
            (map_result
               (fun x : head_of_kind_value_non_null =>
                  @Ok head_of_kind_value_non_null x))
            (meet_head_of_kind_value_non_null join' choices m' env elt1
               elt2))).
Proof.
  intros env non_null1 non_null2.
  apply meet_or_unknown_or_bottom_renamed. intros env' elt1 elt2.
  rewrite meet_head_of_kind_value_non_null_renamed.
  destruct (meet_head_of_kind_value_non_null join choices m env' elt1 elt2)
    as [result | ];
    cbn [option_map]; [ | reflexivity ].
  f_equal. apply map_result_renamed. intros head. reflexivity.
Qed.

Lemma meet_is_null_renamed :
  meet_of_parts_related r (rename_is_null r) (rename_is_null r)
    (fun env is_null1 is_null2 =>
       match is_null1, is_null2 with
       | Not_null, Not_null => Some (Meet_result.Bottom Both_inputs)
       | Not_null, Maybe_null _ => Some (Meet_result.Bottom Left_input)
       | Maybe_null _, Not_null => Some (Meet_result.Bottom Right_input)
       | Maybe_null is_null1, Maybe_null is_null2 =>
         option_map (map_result Maybe_null)
           (meet_relation m env is_null1 is_null2)
       end)
    (fun env is_null1 is_null2 =>
       match is_null1, is_null2 with
       | Not_null, Not_null => Some (Meet_result.Bottom Both_inputs)
       | Not_null, Maybe_null _ => Some (Meet_result.Bottom Left_input)
       | Maybe_null _, Not_null => Some (Meet_result.Bottom Right_input)
       | Maybe_null is_null1, Maybe_null is_null2 =>
         option_map (map_result Maybe_null)
           (meet_relation m' env is_null1 is_null2)
       end).
Proof.
  intros env [ | var1] [ | var2]; cbn [rename_is_null]; try reflexivity.
  rewrite (meet_relation_renamed r m m' Hm).
  destruct (meet_relation m env var1 var2) as [result | ];
    cbn [option_map]; [ | reflexivity ].
  f_equal. apply map_result_renamed. intros var. reflexivity.
Qed.

Lemma meet_head_of_kind_value_renamed :
  meet_of_parts_related r (rename_head_of_kind_value r)
    (rename_head_of_kind_value r) (meet_head_of_kind_value join choices m)
    (meet_head_of_kind_value join' choices m').
Proof.
  intros env [non_null1 is_null1] [non_null2 is_null2].
  cbn [rename_head_of_kind_value meet_head_of_kind_value]. cbv zeta.
  pose proof
    (meet_disjunction_renamed r join join' choices m m' Hjoin Hm
       (fun r =>
          or_unknown_or_bottom_map (rename_head_of_kind_value_non_null r))
       rename_is_null _ _ _ _
       (fun _ : unit => @Bottom head_of_kind_value_non_null)
       (fun _ : unit => Not_null) meet_non_null_renamed meet_is_null_renamed
       eq_refl eq_refl env non_null1 is_null1 No_extensions non_null2
       is_null2 No_extensions)
    as Hdisjunction.
  cbn [rename_variant_extensions] in Hdisjunction.
  rewrite Hdisjunction.
  match goal with
  | |- option_map _ (option_map _ ?call) = _ =>
    destruct call as [result | ]
  end;
    cbn [option_map]; [ | reflexivity ].
  f_equal. apply map_result_renamed.
  intros [[non_null is_null] extensions]. reflexivity.
Qed.

End Values.
