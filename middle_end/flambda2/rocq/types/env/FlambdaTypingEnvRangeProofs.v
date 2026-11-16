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


(* Proofs of the statements of FlambdaTypingEnvRangeSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaPrimitiveProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaRangeSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRangeSpec.
Import ListNotations.

Local Open Scope Z_scope.

Theorem satisfies_in_range_strictly : satisfies_in_range_strictly_statement.
Proof.
  intros tenv statics vars [Hbound [valuation [Hsatisfied [Hagrees _]]]].
  split; [ exact Hbound | ]. exists valuation.
  split; [ exact Hsatisfied | exact Hagrees ].
Qed.

(* ======================================================================== *)
(* What a valuation in range gives                                          *)
(* ======================================================================== *)

Theorem in_range_blocks_have_tags : in_range_blocks_have_tags_statement.
Proof.
  intros statics valuation simple v tag fields [_ Hstatics] Hvaluation Heval
    Hblock.
  assert (Hv : value_in_range v = true).
  { destruct simple as [x | sym | c]; simpl in Heval.
    - exact (Hvaluation x v (lookup_in _ _ _ Heval)).
    - injection Heval as <-. reflexivity.
    - injection Heval as <-. discriminate. }
  destruct (block_of_in_range statics v tag fields) as [Htag Hfields];
    [ | exact Hv | exact Hblock | ].
  - intros sym allocated _ Hlookup.
    exact (Hstatics sym allocated (lookup_in _ _ _ Hlookup)).
  - split; [ exact Htag | ]. split.
    + unfold tag_in_range in Htag. apply andb_true_iff in Htag.
      destruct Htag as [Hlow Hhigh]. apply Z.leb_le in Hlow.
      apply Z.ltb_lt in Hhigh. unfold is_tag. lia.
    + apply forallb_forall. exact Hfields.
Qed.

Theorem agrees_in_terms_has_value : agrees_in_terms_has_value_statement.
Proof.
  intros tenv valuation vars x Hbound Hagrees Hscope.
  destruct (Hbound x Hscope) as [v Hv]. exists v.
  split; [ rewrite (Hagrees x Hscope); exact Hv | exact Hv ].
Qed.

(* ======================================================================== *)
(* Where it holds                                                           *)
(* ======================================================================== *)

Theorem create_satisfies_in_range : create_satisfies_in_range_statement.
Proof.
  intros statics vars. split; [ intros x Hscope; discriminate | ].
  exists []. split; [ | split ].
  - split; [ reflexivity | ]. split.
    + intros n ty binding_time_and_mode v Hlookup. discriminate.
    + apply empty_respected.
  - intros x Hscope. discriminate.
  - intros x v [].
Qed.

Theorem satisfied_strictly_satisfies_in_range
    : satisfied_strictly_satisfies_in_range_statement.
Proof.
  intros tenv statics vars Hbound Hsatisfied Hrange.
  split; [ exact Hbound | ]. exists vars.
  split; [ exact Hsatisfied | ].
  split; [ intros x _; reflexivity | exact Hrange ].
Qed.

(* ======================================================================== *)
(* The environment follows the term                                         *)
(* ======================================================================== *)

Theorem increment_scope_satisfies_in_range
    : increment_scope_satisfies_in_range_statement.
Proof.
  intros tenv statics vars. split; exact (fun Hsatisfies => Hsatisfies).
Qed.

Theorem add_variable_definition_satisfies_in_range
    : add_variable_definition_satisfies_in_range_statement.
Proof.
  intros tenv statics vars x kind v Hwf Hfresh Hkind Hrange Hsat.
  destruct (add_variable_definition_env_satisfies_strictly tenv statics vars
              x kind v Hwf Hfresh Hkind
              (satisfies_in_range_strictly tenv statics vars Hsat))
    as [Hbound' _].
  destruct Hsat as [Hbound [valuation [Hsatisfied [Hagrees Hvaluation]]]].
  split; [ exact Hbound' | ].
  exists ((x, v) :: valuation). split; [ | split ].
  - apply add_variable_definition_satisfied_bind;
      [ exact Hsatisfied | | exact Hkind ].
    exact (well_formed_not_mentioned tenv x Hwf Hfresh).
  - intros y Hy.
    apply (add_variable_definition_in_scope tenv x kind y Hwf) in Hy.
    destruct Hy as [-> | [Hne Hy]].
    + exact (eq_trans (lookup_head x v valuation)
               (eq_sym (lookup_head x v vars))).
    + exact (eq_trans (lookup_other y x v valuation Hne)
               (eq_trans (Hagrees y Hy)
                  (eq_sym (lookup_other y x v vars Hne)))).
  - intros y w [[= <- <-] | Hin];
      [ exact Hrange | exact (Hvaluation y w Hin) ].
Qed.

Theorem add_definitions_of_params_satisfies_in_range
    : add_definitions_of_params_satisfies_in_range_statement.
Proof.
  intros tenv statics vars vars' params. revert tenv vars.
  induction params as [ | [x kind] params IH];
    intros tenv vars vs Hinvariant Hdistinct Hfresh Hkinds Hrange Hbind Hsat.
  - inversion Hkinds; subst. simpl in Hbind. injection Hbind as <-.
    exact Hsat.
  - inversion Hkinds as [ | param v params' vs' Hkind Hrest]; subst.
    simpl in Hbind.
    change (negb (among x (map fst params)) && distinct (map fst params)
            = true) in Hdistinct.
    apply andb_true_iff in Hdistinct. destruct Hdistinct as [Hx Hdistinct].
    apply negb_true_iff in Hx.
    assert (Hfresh_x : Typing_env.mem None tenv (Name_var x) = false).
    { apply Hfresh. left. reflexivity. }
    apply (IH (Typing_env.add_variable_definition tenv x kind
                 Name_mode.normal) ((x, v) :: vars) vs').
    + apply add_variable_definition_invariant; assumption.
    + exact Hdistinct.
    + intros y Hin. rewrite mem_add_variable_definition.
      * apply Hfresh. right. exact Hin.
      * intros ->. apply among_In in Hin. congruence.
    + exact Hrest.
    + intros w Hin. apply Hrange. right. exact Hin.
    + exact Hbind.
    + apply add_variable_definition_satisfies_in_range;
        [ exact (proj1 Hinvariant) | exact Hfresh_x | exact Hkind
        | apply Hrange; left; reflexivity | exact Hsat ].
Qed.

Theorem replace_equation_satisfies_in_range
    : replace_equation_satisfies_in_range_statement.
Proof.
  intros tenv statics vars x ty Hx Hfree Htype
    [Hbound [valuation [Hsatisfied [Hagrees Hvaluation]]]].
  split.
  - intros y Hy. apply Hbound. apply (replace_equation_in_scope tenv x ty y).
    exact Hy.
  - exists valuation. split; [ | split; [ | exact Hvaluation ] ].
    + apply replace_equation_satisfied; [ exact Hsatisfied | ].
      intros v Hv. right. rewrite value_of_name_var in Hv.
      rewrite (Hagrees x Hx) in Hv.
      apply (type_satisfied_coincide statics vars valuation v ty);
        [ | exact (Htype v Hv) ].
      intros y Hin. symmetry. exact (Hagrees y (Hfree y Hin)).
    + intros y Hy. apply Hagrees.
      apply (replace_equation_in_scope tenv x ty y). exact Hy.
Qed.

Theorem add_alias_satisfies_in_range
    : add_alias_satisfies_in_range_statement.
Proof.
  intros tenv statics vars c1 c2 result Hscope Hadd Heval
    [Hbound [valuation [Hsatisfied [Hagrees Hvaluation]]]].
  split.
  - intros y Hy. apply Hbound.
    apply (add_alias_in_scope tenv c1 c2 result y Hadd). exact Hy.
  - exists valuation. split; [ | split; [ | exact Hvaluation ] ].
    + apply (add_alias_satisfied (fun _ => False) statics valuation tenv c1
               c2 result Hsatisfied Hadd).
      rewrite (agrees_eval_simple tenv valuation vars c1 Hagrees),
        (agrees_eval_simple tenv valuation vars c2 Hagrees);
        [ exact Heval | | ];
        intros x ->; apply Hscope; [ right | left ]; reflexivity.
    + intros y Hy. apply Hagrees.
      apply (add_alias_in_scope tenv c1 c2 result y Hadd). exact Hy.
Qed.

Theorem satisfies_in_range_statics_extend
    : satisfies_in_range_statics_extend_statement.
Proof.
  intros tenv statics statics' vars Hextend
    [Hbound [valuation [Hsatisfied [Hagrees Hvaluation]]]].
  split; [ exact Hbound | ]. exists valuation.
  split; [ | split; [ exact Hagrees | exact Hvaluation ] ].
  exact (satisfied_but_statics_extend _ statics statics' valuation tenv
           Hextend Hsatisfied).
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem example_out_of_range : example_out_of_range_statement.
Proof.
  split; [ | split; [ | split; [ | split ] ] ].
  - (* The invariant: the test, and the equations of the levels. *)
    split; [ vm_compute; reflexivity | ].
    intros n ty Hlookup.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    vm_compute in Hlookup.
    repeat (destruct Hlookup as [Hlookup | Hlookup];
            [ injection Hlookup as <- <- | ]); [ .. | contradiction ];
      eexists; vm_compute; reflexivity.
  - apply types_well_formed_b_iff. vm_compute. reflexivity.
  - (* Satisfied, by the valuation that gives [p] the number. *)
    split.
    + intros x Hscope. apply in_scope_in_terms_lookup in Hscope.
      destruct Hscope as [ty [btm [Hlookup Hmode]]].
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
      vm_compute in Hlookup.
      repeat (destruct Hlookup as [Hlookup | Hlookup];
              [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ];
        try (vm_compute in Hmode; discriminate);
        eexists; reflexivity.
    + exists [(example_i, Value_const (Naked_immediate 1));
              (example_p, tagged example_big_number)].
      split.
      * split; [ reflexivity | ]. split; [ | apply empty_respected ].
        intros n ty btm v Hlookup Hvalue.
        apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
        vm_compute in Hlookup.
        repeat (destruct Hlookup as [Hlookup | Hlookup];
                [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ].
        all: cbv in Hvalue; injection Hvalue as <-; right.
        all: try (first
                    [ apply (proj2 (unknown_satisfied _ _ _ Value))
                    | apply (proj2
                               (unknown_satisfied _ _ _
                                  (Naked_number Kind_naked_immediate))) ];
                  reflexivity).
        -- apply (proj2 (is_int_for_scrutinee_satisfied _ _ _
                           (Name_var example_p))).
           exists 1, (tagged example_big_number).
           split; [ reflexivity | ]. split; [ reflexivity | ]. left.
           split; [ eexists; reflexivity | reflexivity ].
        -- apply (proj2 (tag_immediate_satisfied _ _ _
                           (these_naked_immediates [example_big_number]))).
           exists example_big_number. split; [ reflexivity | ].
           apply (proj2 (these_naked_immediates_satisfied _ _ _
                           [example_big_number])).
           eexists. split; [ reflexivity | left; reflexivity ].
      * intros x Hscope. apply in_scope_in_terms_lookup in Hscope.
        destruct Hscope as [ty [btm [Hlookup Hmode]]].
        apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
        vm_compute in Hlookup.
        repeat (destruct Hlookup as [Hlookup | Hlookup];
                [ injection Hlookup as <- <- <- | ]); [ .. | contradiction ];
          try (vm_compute in Hmode; discriminate);
          reflexivity.
  - intros x v [[= <- <-] | []]. reflexivity.
  - (* A valuation that satisfies it gives [p] the number. *)
    intros [_ [valuation [Hsatisfied [Hagrees Hrange]]]].
    assert (Hi :
      lookup example_i valuation = Some (Value_const (Naked_immediate 1))).
    { rewrite (Hagrees example_i); reflexivity. }
    pose proof (find_satisfied_strictly empty_static_env valuation
                  example_out_of_range_env (Name_var example_i) None _ _
                  Hsatisfied eq_refl Hi) as Hrelation.
    apply is_int_for_scrutinee_satisfied in Hrelation.
    destruct Hrelation as [i [w [_ [Hw _]]]].
    pose proof (find_satisfied_strictly empty_static_env valuation
                  example_out_of_range_env (Name_var example_p) None _ w
                  Hsatisfied eq_refl Hw) as Hp.
    change (type_satisfied empty_static_env valuation w
              (tag_immediate (these_naked_immediates [example_big_number])))
      in Hp.
    apply tag_immediate_satisfied in Hp. destruct Hp as [j [-> Hj]].
    apply these_naked_immediates_satisfied in Hj.
    destruct Hj as [j' [[= <-] [<- | []]]].
    pose proof (Hrange example_p _ (lookup_in _ _ _ Hw)) as Hbad.
    vm_compute in Hbad. discriminate.
Qed.
