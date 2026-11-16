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


(* The two passes that unbox the parameters of continuations, and the
   names of a term: that they commute with renaming, and that they
   keep scope. Nothing here needs review.

   The statements are those of FlambdaUnboxParamsRenameSpec.v and of
   FlambdaUnboxVariantsRenameSpec.v, whose Proofs files take them from
   here. They are proved in this file because the validation runs one
   of the passes: FlambdaValidatorEquivarianceProofs.v and
   FlambdaValidateScopeProofs.v need what is proved here for the round
   of FlambdaValidateSpec.v, so this file imports neither. What the
   proofs of renaming use of the first of them, the equations for what
   is known of blocks and of wrapped numbers and three more, is taken
   as premises in the section below, under the names that file gives
   them; the two Proofs files, and that file where it uses the result,
   give the premises. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaBindersSpec FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaProofs FlambdaNormaliseSpec
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBlockLoadSpec
  FlambdaBlockLoadProofs FlambdaUnboxSpec FlambdaValidatorEquivarianceSpec
  FlambdaScopedProofs FlambdaKeepsScopeProofs FlambdaSemanticsSpec
  FlambdaUnusedContsSpec FlambdaCompareSpec FlambdaCompareAlphaSpec
  FlambdaCompareAlphaProofs FlambdaCseSpec FlambdaCseProofs
  FlambdaUnboxParamsSpec FlambdaUnboxParamsProofs
  FlambdaUnboxParamsRenameSpec FlambdaUnboxVariantsSpec
  FlambdaUnboxVariantsProofs FlambdaUnboxVariantsRenameSpec
  FlambdaUnboxLoopsSpec FlambdaEquivarianceTactic.
Import ListNotations.

(* The comparison answers the same when both terms are renamed
   (FlambdaCompareAlphaProofs.expr_eqb_equivariant), here with no names
   paired. *)
Lemma expr_eqb_under_renaming (r : renaming) (e e' : expr) :
  expr_eqb [] [] (rename_expr r e) (rename_expr r e') = expr_eqb [] [] e e'.
Proof. exact (expr_eqb_equivariant r [] [] e e'). Qed.

(* ======================================================================== *)
(* The passes commute with renaming                                         *)
(* ======================================================================== *)

Section Renamed.

(* The equations of FlambdaValidatorEquivarianceProofs.v that the
   proofs use, each under its name there. *)
Hypothesis var_free_b_equivariant : var_free_b_equivariant_statement.

Hypothesis remove_unused_conts_equivariant :
  remove_unused_conts_equivariant_statement.

Hypothesis forget_vars_equivariant :
  ∀ (r : renaming) (xs : list variable) (known : known_blocks),
    forget_vars (map (rename_variable r) xs) (rename_known_blocks r known)
    = rename_known_blocks r (forget_vars xs known).

Hypothesis known_under_let_equivariant :
  ∀ (r : renaming) (known : known_blocks) (x : variable) (d : named),
    known_under_let (rename_known_blocks r known) (rename_variable r x)
      (rename_named r d)
    = rename_known_blocks r (known_under_let known x d).

Hypothesis known_under_statics_equivariant :
  ∀ (r : renaming) (known : known_blocks) (bindings : list static_binding),
    known_under_statics (rename_known_blocks r known)
      (map (rename_static_binding r) bindings)
    = rename_known_blocks r (known_under_statics known bindings).

Hypothesis known_block_of_equivariant :
  ∀ (r : renaming) (known : known_blocks) (s : simple),
    known_block_of (rename_known_blocks r known) (rename_simple r s)
    = option_map (rename_known_block r) (known_block_of known s).

Hypothesis field_loaded_equivariant :
  ∀ (r : renaming) (kind : block_access_kind) (field : Z)
    (block : known_block),
    field_loaded kind field (rename_known_block r block)
    = option_map (rename_simple r) (field_loaded kind field block).

Hypothesis load_known_named_equivariant :
  ∀ (r : renaming) (known : known_blocks) (d : named),
    load_known_named (rename_known_blocks r known) (rename_named r d)
    = rename_named r (load_known_named known d).

Hypothesis forget_wrapped_vars_equivariant :
  ∀ (r : renaming) (xs : list variable) (known : known_numbers),
    forget_wrapped_vars (map (rename_variable r) xs)
      (rename_known_numbers r known)
    = rename_known_numbers r (forget_wrapped_vars xs known).

Hypothesis numbers_under_let_equivariant :
  ∀ (r : renaming) (known : known_numbers) (x : variable) (d : named),
    numbers_under_let (rename_known_numbers r known) (rename_variable r x)
      (rename_named r d)
    = rename_known_numbers r (numbers_under_let known x d).

Hypothesis numbers_under_statics_equivariant :
  ∀ (r : renaming) (known : known_numbers) (bindings : list static_binding),
    numbers_under_statics (rename_known_numbers r known)
      (map (rename_static_binding r) bindings)
    = rename_known_numbers r (numbers_under_statics known bindings).

Hypothesis unwrapped_equivariant :
  ∀ (r : renaming) (known : known_numbers) (how : wrapping)
    (wrapped : simple),
    unwrapped (rename_known_numbers r known) how (rename_simple r wrapped)
    = option_map (rename_simple r) (unwrapped known how wrapped).

Hypothesis unbox_known_named_equivariant :
  ∀ (r : renaming) (known : known_numbers) (d : named),
    unbox_known_named (rename_known_numbers r known) (rename_named r d)
    = rename_named r (unbox_known_named known d).

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* The laws below that the tactic of FlambdaEquivarianceTactic.v proves
   are proved by it. The types of what is known do not say how it is
   renamed: the actions are given here. The premises of the section are
   given to the tactic, and each law of the section once it is proved. *)
#[local] Instance facts_renaming : Renaming_of facts := rename_facts.
#[local] Instance known_blocks_renaming : Renaming_of known_blocks :=
  rename_known_blocks.
#[local] Instance known_block_renaming : Renaming_of known_block :=
  rename_known_block.
#[local] Instance known_numbers_renaming : Renaming_of known_numbers :=
  rename_known_numbers.
#[local] Instance computed_renaming : Renaming_of computed := rename_computed.
#[local] Hint Rewrite var_free_b_equivariant remove_unused_conts_equivariant
  forget_vars_equivariant known_under_let_equivariant
  known_under_statics_equivariant known_block_of_equivariant
  field_loaded_equivariant load_known_named_equivariant
  forget_wrapped_vars_equivariant numbers_under_let_equivariant
  numbers_under_statics_equivariant unwrapped_equivariant
  unbox_known_named_equivariant : equivariance.


Lemma facts_under_let_equivariant (r : renaming) (known : facts)
    (x : variable) (d : named) :
  facts_under_let (rename_facts r known) (rename_variable r x)
    (rename_named r d)
  = rename_facts r (facts_under_let known x d).
Proof. equivariance. Qed.

#[local] Hint Rewrite facts_under_let_equivariant : equivariance.

Lemma facts_under_statics_equivariant (r : renaming) (known : facts)
    (bindings : list static_binding) :
  facts_under_statics (rename_facts r known)
    (map (rename_static_binding r) bindings)
  = rename_facts r (facts_under_statics known bindings).
Proof. equivariance. Qed.

#[local] Hint Rewrite facts_under_statics_equivariant : equivariance.

Lemma facts_under_params_equivariant (r : renaming) (known : facts)
    (params : list variable) :
  facts_under_params (rename_facts r known) (map (rename_variable r) params)
  = rename_facts r (facts_under_params known params).
Proof. equivariance. Qed.

(* ======================================================================== *)
(* What reading a box gives, and the applications                           *)
(* ======================================================================== *)

Theorem projection_of_equivariant : projection_of_equivariant_statement.
Proof.
  intros r known p arg. unfold projection_of.
  destruct (reads_a_box p); [ | reflexivity ].
  change (Prim (Unary p) [rename_simple r arg])
    with (rename_named r (Prim (Unary p) [arg])).
  cbn [rename_facts facts_blocks facts_numbers facts_reads].
  rewrite fold_named_equivariant, load_known_named_equivariant,
    unbox_known_named_equivariant.
  pose proof (computed_by_renamed r (facts_reads known) (Unary p) [arg])
    as Hby.
  cbn [map] in Hby.
  destruct (fold_named (Prim (Unary p) [arg])); try reflexivity;
    (destruct (load_known_named (facts_blocks known) (Prim (Unary p) [arg]));
     try reflexivity;
     (destruct
        (unbox_known_named (facts_numbers known) (Prim (Unary p) [arg]));
      try reflexivity;
      (cbn [rename_named]; rewrite Hby;
       destruct (computed_by (facts_reads known) (Unary p) [arg]);
       reflexivity))).
Qed.

#[local] Hint Rewrite facts_under_params_equivariant
  projection_of_equivariant : equivariance.

Lemma extend_args_equivariant (r : renaming) (known : facts) (i : nat)
    (p : unary_primitive) (args : list simple) :
  extend_args (rename_facts r known) i p (map (rename_simple r) args)
  = option_map (map (rename_simple r)) (extend_args known i p args).
Proof. equivariance. Qed.

#[local] Hint Rewrite extend_args_equivariant : equivariance.

Lemma extend_apply_cont_equivariant (r : renaming) (known : facts)
    (k : continuation) (i : nat) (p : unary_primitive) (ac : apply_cont) :
  extend_apply_cont (rename_facts r known) (rename_continuation r k) i p
    (rename_apply_cont r ac)
  = option_map (rename_apply_cont r) (extend_apply_cont known k i p ac).
Proof. equivariance. Qed.

#[local] Hint Rewrite extend_apply_cont_equivariant : equivariance.

Lemma extend_arms_equivariant (r : renaming) (known : facts)
    (k : continuation) (i : nat) (p : unary_primitive)
    (arms : list (Z * apply_cont)) :
  extend_arms (rename_facts r known) (rename_continuation r k) i p
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = option_map (map (fun arm => (fst arm, rename_apply_cont r (snd arm))))
      (extend_arms known k i p arms).
Proof. equivariance. Qed.

#[local] Hint Rewrite extend_arms_equivariant : equivariance.

Lemma is_bound_renamed_handlers (r : renaming) (k : continuation)
    (handlers : list (continuation * continuation_handler)) :
  is_bound (rename_continuation r k)
    (map (fun '(k', h) => (rename_continuation r k', rename_handler r h))
       handlers)
  = is_bound k handlers.
Proof. equivariance. Qed.

Fixpoint extend_applications_equivariant_aux (r : renaming) (e : expr) :
  ∀ (known : facts) (k : continuation) (i : nat) (p : unary_primitive),
    extend_applications (rename_facts r known) (rename_continuation r k) i p
      (rename_expr r e)
    = option_map (rename_expr r) (extend_applications known k i p e)
with extend_in_handler_equivariant (r : renaming) (h : continuation_handler) :
  ∀ (known : facts) (k : continuation) (i : nat) (p : unary_primitive),
    extend_in_handler (rename_facts r known) (rename_continuation r k) i p
      (rename_handler r h)
    = option_map (rename_handler r) (extend_in_handler known k i p h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k0 h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known k i p; cbn [rename_expr extend_applications].
    + rewrite facts_under_let_equivariant,
        extend_applications_equivariant_aux.
      destruct (extend_applications (facts_under_let known x d) k i p body);
        reflexivity.
    + rewrite facts_under_statics_equivariant,
        extend_applications_equivariant_aux.
      destruct
        (extend_applications (facts_under_statics known bindings) k i p body);
        reflexivity.
    + rewrite extend_in_handler_equivariant.
      destruct (extend_in_handler known k i p h) as [h' | ]; simpl;
        [ | reflexivity ].
      rewrite rename_continuation_eqb.
      destruct (Pos.eqb k0 k); [ reflexivity | ].
      rewrite extend_applications_equivariant_aux.
      destruct (extend_applications known k i p body); reflexivity.
    + rewrite is_bound_renamed_handlers.
      destruct (is_bound k handlers); [ reflexivity | ].
      assert (Hhandlers :
        all_some
          (map
             (fun '(k', h) =>
                let* h :=
                  extend_in_handler (rename_facts r known)
                    (rename_continuation r k) i p h
                in
                Some (k', h))
             (map
                (fun '(k', h) => (rename_continuation r k', rename_handler r h))
                handlers))
        = option_map
            (map
               (fun '(k', h) =>
                  (rename_continuation r k', rename_handler r h)))
            (all_some
               (map
                  (fun '(k', h) =>
                     let* h := extend_in_handler known k i p h in
                     Some (k', h))
                  handlers))).
      { induction handlers as [ | [k1 h1] handlers IH]; simpl;
          [ reflexivity | ].
        rewrite (extend_in_handler_equivariant r h1 known k i p).
        destruct (extend_in_handler known k i p h1); simpl; [ | reflexivity ].
        rewrite IH.
        destruct
          (all_some
             (map
                (fun '(k', h) =>
                   let* h := extend_in_handler known k i p h in
                   Some (k', h))
                handlers));
          reflexivity. }
      rewrite Hhandlers.
      destruct
        (all_some
           (map
              (fun '(k', h) =>
                 let* h := extend_in_handler known k i p h in
                 Some (k', h))
              handlers));
        simpl; [ | reflexivity ].
      rewrite extend_applications_equivariant_aux.
      destruct (extend_applications known k i p body); reflexivity.
    + cbn [rename_apply ap_continuation ap_exn_continuation].
      rewrite !rename_continuation_eqb.
      destruct
        (Pos.eqb (ap_continuation ap) k || Pos.eqb (ap_exn_continuation ap) k);
        reflexivity.
    + rewrite extend_apply_cont_equivariant.
      destruct (extend_apply_cont known k i p ac); reflexivity.
    + rewrite extend_arms_equivariant.
      destruct (extend_arms known k i p arms); reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. intros known k i p.
    cbn [rename_handler extend_in_handler].
    rewrite facts_under_params_equivariant,
      extend_applications_equivariant_aux.
    destruct
      (extend_applications (facts_under_params known params) k i p handler);
      reflexivity.
Qed.

Theorem extend_applications_equivariant :
  extend_applications_equivariant_statement.
Proof.
  intros r known k i p e. apply extend_applications_equivariant_aux.
Qed.

#[local] Hint Rewrite is_bound_renamed_handlers
  extend_applications_equivariant : equivariance.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Lemma position_renamed (r : renaming) (x : variable) (xs : list variable) :
  position (rename_variable r x) (map (rename_variable r) xs) = position x xs.
Proof. equivariance. Qed.

#[local] Hint Rewrite position_renamed : equivariance.

Lemma may_take_renamed (r : renaming) (params : list variable)
    (metadata : handler_metadata) (f : variable) :
  may_take (map (rename_variable r) params) metadata (rename_variable r f)
  = may_take params metadata f.
Proof. equivariance. Qed.

#[local] Hint Rewrite may_take_renamed : equivariance.

Theorem unbox_params_knowing_equivariant :
  unbox_params_knowing_equivariant_statement.
Proof. equivariance. Qed.

Theorem unbox_known_params_equivariant :
  unbox_known_params_equivariant_statement.
Proof.
  intros r e. exact (unbox_params_knowing_equivariant r no_facts e).
Qed.

(* ======================================================================== *)
(* The applications, extended by a function that commutes with renaming     *)
(* ======================================================================== *)

Section By_renamed.

Variable r : renaming.
Variable extend extend' : facts -> list simple -> option (list simple).

Hypothesis extend_renamed :
  ∀ (known : facts) (args : list simple),
    extend' (rename_facts r known) (map (rename_simple r) args)
    = option_map (map (rename_simple r)) (extend known args).

#[local] Hint Rewrite unbox_params_knowing_equivariant
  unbox_known_params_equivariant extend_renamed : equivariance.

Lemma extend_apply_cont_by_equivariant (known : facts) (k : continuation)
    (ac : apply_cont) :
  extend_apply_cont_by extend' (rename_facts r known)
    (rename_continuation r k) (rename_apply_cont r ac)
  = option_map (rename_apply_cont r) (extend_apply_cont_by extend known k ac).
Proof. equivariance. Qed.

#[local] Hint Rewrite extend_apply_cont_by_equivariant : equivariance.

Lemma extend_arms_by_equivariant (known : facts) (k : continuation)
    (arms : list (Z * apply_cont)) :
  extend_arms_by extend' (rename_facts r known) (rename_continuation r k)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = option_map (map (fun arm => (fst arm, rename_apply_cont r (snd arm))))
      (extend_arms_by extend known k arms).
Proof. equivariance. Qed.

Fixpoint extend_applications_by_equivariant_aux (e : expr) :
  ∀ (known : facts) (k : continuation),
    extend_applications_by extend' (rename_facts r known)
      (rename_continuation r k) (rename_expr r e)
    = option_map (rename_expr r) (extend_applications_by extend known k e)
with extend_in_handler_by_equivariant (h : continuation_handler) :
  ∀ (known : facts) (k : continuation),
    extend_in_handler_by extend' (rename_facts r known)
      (rename_continuation r k) (rename_handler r h)
    = option_map (rename_handler r) (extend_in_handler_by extend known k h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k0 h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known k; cbn [rename_expr extend_applications_by].
    + rewrite facts_under_let_equivariant,
        extend_applications_by_equivariant_aux.
      destruct
        (extend_applications_by extend (facts_under_let known x d) k body);
        reflexivity.
    + rewrite facts_under_statics_equivariant,
        extend_applications_by_equivariant_aux.
      destruct
        (extend_applications_by extend (facts_under_statics known bindings) k
           body);
        reflexivity.
    + rewrite extend_in_handler_by_equivariant.
      destruct (extend_in_handler_by extend known k h) as [h' | ]; simpl;
        [ | reflexivity ].
      rewrite rename_continuation_eqb.
      destruct (Pos.eqb k0 k); [ reflexivity | ].
      rewrite extend_applications_by_equivariant_aux.
      destruct (extend_applications_by extend known k body); reflexivity.
    + rewrite is_bound_renamed_handlers.
      destruct (is_bound k handlers); [ reflexivity | ].
      assert (Hhandlers :
        all_some
          (map
             (fun '(k', h) =>
                let* h :=
                  extend_in_handler_by extend' (rename_facts r known)
                    (rename_continuation r k) h
                in
                Some (k', h))
             (map
                (fun '(k', h) => (rename_continuation r k', rename_handler r h))
                handlers))
        = option_map
            (map
               (fun '(k', h) =>
                  (rename_continuation r k', rename_handler r h)))
            (all_some
               (map
                  (fun '(k', h) =>
                     let* h := extend_in_handler_by extend known k h in
                     Some (k', h))
                  handlers))).
      { induction handlers as [ | [k1 h1] handlers IH]; simpl;
          [ reflexivity | ].
        rewrite (extend_in_handler_by_equivariant h1 known k).
        destruct (extend_in_handler_by extend known k h1); simpl;
          [ | reflexivity ].
        rewrite IH.
        destruct
          (all_some
             (map
                (fun '(k', h) =>
                   let* h := extend_in_handler_by extend known k h in
                   Some (k', h))
                handlers));
          reflexivity. }
      rewrite Hhandlers.
      destruct
        (all_some
           (map
              (fun '(k', h) =>
                 let* h := extend_in_handler_by extend known k h in
                 Some (k', h))
              handlers));
        simpl; [ | reflexivity ].
      rewrite extend_applications_by_equivariant_aux.
      destruct (extend_applications_by extend known k body); reflexivity.
    + cbn [rename_apply ap_continuation ap_exn_continuation].
      rewrite !rename_continuation_eqb.
      destruct
        (Pos.eqb (ap_continuation ap) k || Pos.eqb (ap_exn_continuation ap) k);
        reflexivity.
    + rewrite extend_apply_cont_by_equivariant.
      destruct (extend_apply_cont_by extend known k ac); reflexivity.
    + rewrite extend_arms_by_equivariant.
      destruct (extend_arms_by extend known k arms); reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. intros known k.
    cbn [rename_handler extend_in_handler_by].
    rewrite facts_under_params_equivariant,
      extend_applications_by_equivariant_aux.
    destruct
      (extend_applications_by extend (facts_under_params known params) k
         handler);
      reflexivity.
Qed.

End By_renamed.

Theorem extend_applications_by_equivariant :
  extend_applications_by_equivariant_statement.
Proof.
  intros r extend extend' Hextend known k e.
  exact (extend_applications_by_equivariant_aux r extend extend' Hextend e
    known k).
Qed.

(* ======================================================================== *)
(* The argument for a parameter that is read under a test                   *)
(* ======================================================================== *)

Lemma lookup_known_var_renamed (r : renaming) (known : facts)
    (y : variable) :
  lookup (rename_variable r y)
    (known_vars (facts_blocks (rename_facts r known)))
  = option_map (rename_known_block r)
      (lookup y (known_vars (facts_blocks known))).
Proof. exact (known_block_of_equivariant r (facts_blocks known) (Var y)). Qed.

Lemma no_value_on_renamed (r : renaming) (known : facts)
    (p : unary_primitive) (arg : simple) :
  no_value_on (rename_facts r known) p (rename_simple r arg)
  = no_value_on known p arg.
Proof.
  destruct p, arg as [y | sym | c]; try reflexivity.
  cbn [no_value_on rename_simple]. unfold is_bound.
  rewrite lookup_known_var_renamed.
  destruct (lookup y (known_vars (facts_blocks known))); reflexivity.
Qed.

Lemma made_field_equivariant (r : renaming) (known : facts)
    (p : unary_primitive) (kind : flambda_kind) (arg : simple) :
  made_field (rename_facts r known) p kind (rename_simple r arg)
  = option_map (rename_simple r) (made_field known p kind arg).
Proof.
  destruct p as [access mut field | | | | | | | | | | | | | | ];
    try reflexivity; try (destruct arg; reflexivity).
  destruct mut, arg as [y | sym | c]; try reflexivity;
    (cbn [made_field rename_simple]; rewrite lookup_known_var_renamed;
     destruct (lookup y (known_vars (facts_blocks known))) as [block | ];
     simpl; [ | reflexivity ];
     rewrite field_loaded_equivariant;
     destruct (field_loaded access field block); simpl; [ | reflexivity ];
     destruct (kind_eqb kind Value); reflexivity).
Qed.

Lemma made_number_equivariant (r : renaming) (known : facts)
    (p : unary_primitive) (kind : flambda_kind) (arg : simple) :
  made_number (rename_facts r known) p kind (rename_simple r arg)
  = option_map (rename_simple r) (made_number known p kind arg).
Proof.
  destruct p as [ | | | | | | | | | | boxed | | | | ];
    try reflexivity; try (destruct arg; reflexivity);
    (destruct arg as [y | sym | c]; try reflexivity);
    unfold made_number; cbn [rename_facts facts_numbers rename_simple];
    change (Var (rename_variable r y)) with (rename_simple r (Var y));
    rewrite unwrapped_equivariant.
  - destruct (unwrapped (facts_numbers known) (Boxed boxed) (Var y)); simpl;
      [ | reflexivity ].
    destruct (kind_eqb kind (unboxed_kind boxed)); reflexivity.
  - destruct (unwrapped (facts_numbers known) Tagged (Var y)); simpl;
      [ | reflexivity ].
    destruct (kind_eqb kind (Naked_number Kind_naked_immediate));
      reflexivity.
Qed.

#[local] Hint Rewrite lookup_known_var_renamed no_value_on_renamed
  made_field_equivariant made_number_equivariant : equivariance.

Lemma certain_projection_equivariant (r : renaming) (known : facts)
    (p : unary_primitive) (kind : flambda_kind) (arg : simple) :
  certain_projection (rename_facts r known) p kind (rename_simple r arg)
  = option_map (rename_simple r) (certain_projection known p kind arg).
Proof. equivariance. Qed.

#[local] Hint Rewrite certain_projection_equivariant : equivariance.

Lemma lookup_rename_immediates (r : renaming) (consts : list (variable * Z))
    (t : variable) :
  lookup (rename_variable r t) (rename_immediates r consts) = lookup t consts.
Proof. equivariance. Qed.

#[local] Hint Rewrite lookup_rename_immediates : equivariance.

Lemma forget_immediates_equivariant (r : renaming) (bound : list variable)
    (consts : list (variable * Z)) :
  forget_immediates (map (rename_variable r) bound)
    (rename_immediates r consts)
  = rename_immediates r (forget_immediates bound consts).
Proof. equivariance. Qed.

#[local] Hint Rewrite forget_immediates_equivariant : equivariance.

Lemma only_of_equivariant (r : renaming) (tests : list variable)
    (consts : list (variable * Z)) :
  only_of (map (rename_variable r) tests) (rename_immediates r consts)
  = rename_immediates r (only_of tests consts).
Proof. equivariance. Qed.

#[local] Hint Rewrite only_of_equivariant : equivariance.

Theorem take_arms_equivariant : take_arms_equivariant_statement.
Proof. equivariance. Qed.

#[local] Hint Rewrite take_arms_equivariant : equivariance.

Lemma known_immediates_equivariant (r : renaming) (params : list variable) :
  ∀ args : list simple,
    known_immediates (map (rename_variable r) params)
      (map (rename_simple r) args)
    = rename_immediates r (known_immediates params args).
Proof. equivariance. Qed.

Lemma never_reads_renamed (r : renaming) (consts : list (variable * Z))
    (handler without : expr) :
  never_reads (rename_immediates r consts) (rename_expr r handler)
    (rename_expr r without)
  = never_reads consts handler without.
Proof.
  unfold never_reads.
  rewrite !take_arms_equivariant, !remove_unused_conts_equivariant,
    expr_eqb_under_renaming.
  reflexivity.
Qed.

#[local] Hint Rewrite known_immediates_equivariant never_reads_renamed :
  equivariance.

Theorem extend_args_certain_equivariant :
  extend_args_certain_equivariant_statement.
Proof. equivariance. Qed.

#[local] Hint Rewrite extend_args_certain_equivariant : equivariance.

(* ======================================================================== *)
(* The first read of a parameter                                            *)
(* ======================================================================== *)

Lemma param_read_renamed (r : renaming) (wanted : unary_primitive -> bool)
    (params bound : list variable) (f : variable) (d : named) :
  param_read wanted (map (rename_variable r) params)
    (map (rename_variable r) bound) (rename_variable r f) (rename_named r d)
  = param_read wanted params bound f d.
Proof. equivariance. Qed.

Lemma load_read_renamed (r : renaming) (d : named) :
  load_read (rename_named r d)
  = option_map (fun found => (rename_variable r (fst found), snd found))
      (load_read d).
Proof. equivariance. Qed.

Fixpoint first_read_equivariant_aux (r : renaming) (e : expr) :
  ∀ (wanted : unary_primitive -> bool) (params bound : list variable),
    first_read wanted (map (rename_variable r) params)
      (map (rename_variable r) bound) (rename_expr r e)
    = option_map (rename_read r) (first_read wanted params bound e).
Proof.
  destruct e
    as [ y kind d body | bindings body | k [handler_params metadata handler]
           body
       | handlers body | ap | ac | scrutinee arms | ];
    intros wanted params bound; cbn [rename_expr rename_handler first_read];
    try reflexivity.
  - rewrite param_read_renamed.
    change (rename_variable r y :: map (rename_variable r) bound)
      with (map (rename_variable r) (y :: bound)).
    destruct (param_read wanted params bound y d) as [[p i] | ].
    + rewrite load_read_renamed.
      destruct (load_read d) as [[x field] | ]; cbn [option_map fst snd];
        [ | reflexivity ].
      change [rename_variable r x] with (map (rename_variable r) [x]).
      rewrite first_read_equivariant_aux.
      destruct (first_read (loads_above field) [x] (y :: bound) body);
        reflexivity.
    + rewrite first_read_equivariant_aux.
      destruct (first_read wanted params (y :: bound) body); reflexivity.
  - rewrite first_read_equivariant_aux.
    destruct (first_read wanted params bound body); [ reflexivity | ].
    cbn [option_map]. rewrite <- map_app, first_read_equivariant_aux.
    destruct (first_read wanted params (handler_params ++ bound) handler);
      reflexivity.
Qed.

Theorem first_read_equivariant : first_read_equivariant_statement.
Proof. intros r wanted params bound e. apply first_read_equivariant_aux. Qed.

Lemma next_read_equivariant (r : renaming) (params : list variable)
    (handler : expr) :
  next_read (map (rename_variable r) params) (rename_expr r handler)
  = option_map (rename_read r) (next_read params handler).
Proof.
  unfold next_read.
  pose proof
    (fun wanted => first_read_equivariant_aux r handler wanted params [])
    as Hfirst.
  cbn [map] in Hfirst. rewrite !Hfirst.
  destruct (first_read reads_tag params [] handler); [ reflexivity | ].
  destruct (first_read reads_constant_constructor params [] handler);
    [ reflexivity | ].
  destruct (first_read reads_is_int params [] handler); reflexivity.
Qed.

Lemma head_read_equivariant (r : renaming) (params : list variable)
    (e : expr) :
  head_read (map (rename_variable r) params) (rename_expr r e)
  = option_map (rename_read r) (head_read params e).
Proof.
  destruct e as [ f kind d rest | | | | | | | ]; try reflexivity.
  cbn [rename_expr head_read].
  pose proof (param_read_renamed r reads_any params [] f d) as Hread.
  cbn [map] in Hread. rewrite Hread.
  destruct (param_read reads_any params [] f d) as [[p i] | ]; reflexivity.
Qed.

Lemma tested_is_int_renamed (r : renaming) (params : list variable)
    (i : nat) (handler : expr) :
  tested_is_int (map (rename_variable r) params) i (rename_expr r handler)
  = tested_is_int params i handler.
Proof.
  unfold tested_is_int. rewrite nth_error_map.
  destruct (nth_error params i) as [param | ]; [ | reflexivity ].
  cbn [option_map].
  pose proof (first_read_equivariant_aux r handler reads_is_int [param] [])
    as Hfirst.
  cbn [map] in Hfirst. rewrite Hfirst.
  destruct (first_read reads_is_int [param] [] handler); reflexivity.
Qed.

#[local] Hint Rewrite load_read_renamed first_read_equivariant
  next_read_equivariant head_read_equivariant tested_is_int_renamed :
  equivariance.

(* The tests that the pass made, with their variables renamed. *)
Definition rename_made (r : renaming) (made : list (variable * nat))
    : list (variable * nat) :=
  map (fun entry => (rename_variable r (fst entry), snd entry)) made.

Lemma tests_of_renamed (r : renaming) (i : nat)
    (made : list (variable * nat)) (above : list (nat * list nat)) :
  tests_of i (rename_made r made) above
  = map (rename_variable r) (tests_of i made above).
Proof. equivariance. Qed.

(* The tags that the applications of a continuation give are those
   that the applications of the renamed expression give, where the
   function that finds a tag answers the same on renamed arguments. *)
Section Tags_renamed.

Variable r : renaming.
Variable tag_of tag_of' : facts -> list simple -> list Z.

Hypothesis tag_of_renamed :
  ∀ (known : facts) (args : list simple),
    tag_of' (rename_facts r known) (map (rename_simple r) args)
    = tag_of known args.

#[local] Hint Rewrite tests_of_renamed tag_of_renamed : equivariance.

Lemma tags_at_apply_cont_renamed (known : facts) (k : continuation)
    (ac : apply_cont) :
  tags_at_apply_cont tag_of' (rename_facts r known) (rename_continuation r k)
    (rename_apply_cont r ac)
  = tags_at_apply_cont tag_of known k ac.
Proof. equivariance. Qed.

#[local] Hint Rewrite tags_at_apply_cont_renamed : equivariance.

Lemma tags_given_renamed (e : expr) :
  ∀ (known : facts) (k : continuation),
    tags_given tag_of' (rename_facts r known) (rename_continuation r k)
      (rename_expr r e)
    = tags_given tag_of known k e.
Proof. equivariance. Qed.

End Tags_renamed.

Lemma tag_read_renamed (r : renaming) (tests params : list variable)
    (handler without : expr) (i : nat) (known : facts) (args : list simple) :
  tag_read (map (rename_variable r) tests) (map (rename_variable r) params)
    (rename_expr r handler) (rename_expr r without) i (rename_facts r known)
    (map (rename_simple r) args)
  = tag_read tests params handler without i known args.
Proof. equivariance. Qed.

(* Whether blocks of more than one tag may come to a load is answered
   the same of the renamed term. *)
Lemma loads_of_two_tags_renamed (r : renaming) (known : facts)
    (k : continuation) (tests params : list variable) (handler body : expr)
    (found : read) :
  loads_of_two_tags (rename_facts r known) (rename_continuation r k)
    (map (rename_variable r) tests) (map (rename_variable r) params)
    (rename_expr r handler) (rename_expr r body) (rename_read r found)
  = loads_of_two_tags known k tests params handler body found.
Proof.
  unfold loads_of_two_tags.
  cbn [rename_read read_primitive read_position read_without].
  rewrite
    (tags_given_renamed r
       (tag_read tests params handler (read_without found)
          (read_position found))
       (tag_read (map (rename_variable r) tests)
          (map (rename_variable r) params) (rename_expr r handler)
          (rename_expr r (read_without found)) (read_position found))
       (tag_read_renamed r tests params handler (read_without found)
          (read_position found))
       body known k).
  reflexivity.
Qed.

#[local] Hint Rewrite tag_read_renamed loads_of_two_tags_renamed :
  equivariance.

Lemma lets_in_renamed (r : renaming) (e : expr) :
  lets_in (rename_expr r e) = lets_in e.
Proof. equivariance. Qed.

#[local] Hint Rewrite lets_in_renamed : equivariance.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The parameters, the metadata, the handler and the body that a step
   gives, renamed. *)
(* The reads shared are those of the renamed expression. *)
Lemma reads_same_renamed (r : renaming) (p : unary_primitive) (x : variable)
    (d : named) :
  reads_same p (rename_variable r x) (rename_named r d) = reads_same p x d.
Proof. equivariance. Qed.

#[local] Hint Rewrite reads_same_renamed : equivariance.

Lemma read_of_same_renamed (r : renaming) (f : variable)
    (same : list variable) (d : named) :
  read_of_same (rename_variable r f) (map (rename_variable r) same)
    (rename_named r d)
  = rename_named r (read_of_same f same d).
Proof. equivariance. Qed.

#[local] Hint Rewrite read_of_same_renamed : equivariance.

Lemma share_reads_renamed (r : renaming) (p : unary_primitive)
    (e : expr) :
  ∀ (x f : variable) (same : list variable),
    share_reads p (rename_variable r x) (rename_variable r f)
      (map (rename_variable r) same) (rename_expr r e)
    = rename_expr r (share_reads p x f same e).
Proof. equivariance. Qed.

#[local] Hint Rewrite share_reads_renamed : equivariance.

Lemma with_reads_shared_renamed (r : renaming) (share : bool)
    (params : list variable) (found : read) :
  with_reads_shared share (map (rename_variable r) params)
    (rename_read r found)
  = rename_read r (with_reads_shared share params found).
Proof. equivariance. Qed.

#[local] Hint Rewrite with_reads_shared_renamed : equivariance.

(* Whether the reads are shared is the same of the renamed term. *)
Lemma shares_reads_renamed (r : renaming) (known : facts) (k : continuation)
    (tests params : list variable) (handler body : expr) (found : read) :
  shares_reads (rename_facts r known) (rename_continuation r k)
    (map (rename_variable r) tests) (map (rename_variable r) params)
    (rename_expr r handler) (rename_expr r body) (rename_read r found)
  = shares_reads known k tests params handler body found.
Proof. equivariance. Qed.

Definition rename_step (r : renaming)
    (step : list variable * handler_metadata * expr * expr)
    : list variable * handler_metadata * expr * expr :=
  let '(params, metadata, handler, body) := step in
  (map (rename_variable r) params, metadata, rename_expr r handler,
   rename_expr r body).

Lemma unbox_read_equivariant (r : renaming)
    (extend extend' : facts -> list simple -> option (list simple))
    (known : facts) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr) (found : read) :
  (∀ (known : facts) (args : list simple),
    extend' (rename_facts r known) (map (rename_simple r) args)
    = option_map (map (rename_simple r)) (extend known args)) ->
  unbox_read extend' (rename_facts r known) (rename_continuation r k)
    (map (rename_variable r) params) metadata (rename_expr r handler)
    (rename_expr r body) (rename_read r found)
  = option_map (rename_step r)
      (unbox_read extend known k params metadata handler body found).
Proof.
  intros Hextend. unfold unbox_read.
  cbn [rename_read read_variable read_kind read_without].
  rewrite may_take_renamed, var_free_b_equivariant.
  destruct
    (may_take params metadata (read_variable found)
     && negb (var_free_b (read_variable found) handler));
    [ | reflexivity ].
  rewrite (extend_applications_by_equivariant r extend extend' Hextend).
  destruct (extend_applications_by extend known k body); simpl;
    [ rewrite map_app; reflexivity | reflexivity ].
Qed.

Lemma head_step_equivariant (r : renaming) (known : facts)
    (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler body : expr) (found : read) (share : bool) :
  unbox_read
    (fun known =>
       extend_args known (read_position (rename_read r found))
         (read_primitive (rename_read r found)))
    (rename_facts r known) (rename_continuation r k)
    (map (rename_variable r) params) metadata (rename_expr r handler)
    (rename_expr r body) (rename_read r (with_reads_shared share params found))
  = option_map (rename_step r)
      (unbox_read
         (fun known =>
            extend_args known (read_position found) (read_primitive found))
         known k params metadata handler body
         (with_reads_shared share params found)).
Proof.
  exact (unbox_read_equivariant r _ _ known k params metadata handler body
    (with_reads_shared share params found)
    (fun known args =>
       extend_args_equivariant r known (read_position found)
         (read_primitive found) args)).
Qed.

Lemma certain_step_equivariant (r : renaming) (known : facts)
    (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler body : expr) (found : read) (variant : bool)
    (tests : list variable) :
  unbox_read
    (fun known =>
       extend_args_certain variant (map (rename_variable r) tests)
         (map (rename_variable r) params) (rename_expr r handler)
         (read_without (rename_read r found)) known
         (read_position (rename_read r found))
         (read_primitive (rename_read r found))
         (read_kind (rename_read r found)))
    (rename_facts r known) (rename_continuation r k)
    (map (rename_variable r) params) metadata (rename_expr r handler)
    (rename_expr r body) (rename_read r found)
  = option_map (rename_step r)
      (unbox_read
         (fun known =>
            extend_args_certain variant tests params handler
              (read_without found) known (read_position found)
              (read_primitive found) (read_kind found))
         known k params metadata handler body found).
Proof.
  exact (unbox_read_equivariant r _ _ known k params metadata handler body
    found
    (fun known args =>
       extend_args_certain_equivariant r variant tests params handler
         (read_without found) known (read_position found)
         (read_primitive found) (read_kind found) args)).
Qed.

Lemma unbox_reads_equivariant (r : renaming) (fuel : nat) :
  ∀ (known : facts) (k : continuation) (made : list (variable * nat))
    (above : list (nat * list nat)) (params : list variable)
    (metadata : handler_metadata) (handler body : expr),
    unbox_reads fuel (rename_facts r known) (rename_continuation r k)
      (rename_made r made) above (map (rename_variable r) params) metadata
      (rename_expr r handler) (rename_expr r body)
    = rename_expr r
        (unbox_reads fuel known k made above params metadata handler body).
Proof.
  induction fuel as [ | fuel IH];
    intros known k made above params metadata handler body;
    cbn [unbox_reads]; [ reflexivity | ].
  rewrite next_read_equivariant, head_read_equivariant.
  destruct (next_read params handler) as [found0 | ]; cbn [option_map];
    [ | reflexivity ].
  change (read_position (rename_read r found0)) with (read_position found0).
  rewrite (tests_of_renamed r (read_position found0) made above),
    shares_reads_renamed.
  set (share :=
    shares_reads known k (tests_of (read_position found0) made above) params
      handler body found0).
  rewrite with_reads_shared_renamed.
  set (found := with_reads_shared share params found0).
  change (read_position (rename_read r found)) with (read_position found).
  change (read_primitive (rename_read r found)) with (read_primitive found).
  change (read_kind (rename_read r found)) with (read_kind found).
  change (read_variable (rename_read r found))
    with (rename_variable r (read_variable found)).
  rewrite tested_is_int_renamed, tests_of_renamed, length_map.
  rewrite loads_of_two_tags_renamed.
  destruct
    (loads_of_two_tags known k (tests_of (read_position found) made above)
       params handler body found);
    [ reflexivity | ].
  set (made' :=
    if reads_test (read_primitive found)
    then (read_variable found, read_position found) :: made
    else made).
  assert (Hmade :
    (if reads_test (read_primitive found)
     then
       (rename_variable r (read_variable found), read_position found)
       :: rename_made r made
     else rename_made r made)
    = rename_made r made').
  { unfold made'. destruct (reads_test (read_primitive found)); reflexivity. }
  rewrite Hmade. clear Hmade.
  set (above' :=
    (length params, length params :: boxes_of (read_position found) above)
    :: above).
  set (variant := tested_is_int params (read_position found) handler).
  set (tests := tests_of (read_position found) made above).
  assert (Hcertain :
    match
      unbox_read
        (fun known =>
           extend_args_certain variant (map (rename_variable r) tests)
             (map (rename_variable r) params) (rename_expr r handler)
             (read_without (rename_read r found)) known
             (read_position found) (read_primitive found) (read_kind found))
        (rename_facts r known) (rename_continuation r k)
        (map (rename_variable r) params) metadata (rename_expr r handler)
        (rename_expr r body) (rename_read r found)
    with
    | Some (params0, metadata0, handler0, body0) =>
      unbox_reads fuel (rename_facts r known) (rename_continuation r k)
        (rename_made r made') above' params0 metadata0 handler0 body0
    | None =>
      Let_cont_non_recursive (rename_continuation r k)
        (Handler (map (rename_variable r) params) metadata
           (rename_expr r handler))
        (rename_expr r body)
    end
    = rename_expr r
        match
          unbox_read
            (fun known =>
               extend_args_certain variant tests params handler
                 (read_without found) known (read_position found)
                 (read_primitive found) (read_kind found))
            known k params metadata handler body found
        with
        | Some (params0, metadata0, handler0, body0) =>
          unbox_reads fuel known k made' above' params0 metadata0 handler0
            body0
        | None =>
          Let_cont_non_recursive k (Handler params metadata handler) body
        end).
  { pose proof
      (certain_step_equivariant r known k params metadata handler body found
         variant tests)
      as Hstep.
    change (read_position (rename_read r found)) with (read_position found)
      in Hstep.
    change (read_primitive (rename_read r found))
      with (read_primitive found) in Hstep.
    change (read_kind (rename_read r found)) with (read_kind found) in Hstep.
    rewrite Hstep.
    destruct
      (unbox_read
         (fun known =>
            extend_args_certain variant tests params handler
              (read_without found) known (read_position found)
              (read_primitive found) (read_kind found))
         known k params metadata handler body found)
      as [[[[params' metadata'] handler'] body'] | ];
      cbn [option_map rename_step]; [ apply IH | reflexivity ]. }
  destruct (head_read params handler) as [head | ]; cbn [option_map];
    [ | exact Hcertain ].
  change (read_variable (rename_read r head))
    with (rename_variable r (read_variable head)).
  rewrite rename_variable_eqb.
  destruct (Pos.eqb (read_variable head) (read_variable found));
    [ | exact Hcertain ].
  rewrite with_reads_shared_renamed, head_step_equivariant.
  destruct
    (unbox_read
       (fun known =>
          extend_args known (read_position head) (read_primitive head))
       known k params metadata handler body
       (with_reads_shared share params head))
    as [[[[params' metadata'] handler'] body'] | ];
    cbn [option_map rename_step]; [ apply IH | reflexivity ].
Qed.

Fixpoint unbox_variants_knowing_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : facts,
    unbox_variants_knowing (rename_facts r known) (rename_expr r e)
    = rename_expr r (unbox_variants_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap
       | ac | scrutinee arms | ];
    intros known; cbn [rename_expr unbox_variants_knowing]; try reflexivity.
  - rewrite facts_under_let_equivariant,
      unbox_variants_knowing_equivariant_aux.
    reflexivity.
  - rewrite facts_under_statics_equivariant,
      unbox_variants_knowing_equivariant_aux.
    reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler unbox_variants_knowing].
    rewrite facts_under_params_equivariant,
      !unbox_variants_knowing_equivariant_aux, lets_in_renamed.
    exact (unbox_reads_equivariant r _ _ _ [] [] _ _ _ _).
  - rewrite unbox_variants_knowing_equivariant_aux. reflexivity.
Qed.

Theorem unbox_variants_knowing_equivariant :
  unbox_variants_knowing_equivariant_statement.
Proof. intros r known e. apply unbox_variants_knowing_equivariant_aux. Qed.

Theorem unbox_variant_params_equivariant :
  unbox_variant_params_equivariant_statement.
Proof.
  intros r e. exact (unbox_variants_knowing_equivariant_aux r e no_facts).
Qed.

#[local] Hint Rewrite shares_reads_renamed head_step_equivariant
  certain_step_equivariant unbox_reads_equivariant
  unbox_variants_knowing_equivariant unbox_variant_params_equivariant :
  equivariance.

(* ======================================================================== *)
(* The reads of a loop                                                      *)
(* ======================================================================== *)

Lemma on_spine_renamed (r : renaming) (x : variable) (e : expr) :
  on_spine (rename_variable r x) (rename_expr r e) = on_spine x e.
Proof. equivariance. Qed.

Lemma unbox_loop_read_equivariant (r : renaming) (known : facts)
    (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler body : expr) (found : read) :
  unbox_loop_read (rename_facts r known) (rename_continuation r k)
    (map (rename_variable r) params) metadata (rename_expr r handler)
    (rename_expr r body) (rename_read r found)
  = option_map (rename_step r)
      (unbox_loop_read known k params metadata handler body found).
Proof.
  unfold unbox_loop_read.
  cbn [rename_read read_variable read_kind read_primitive read_position
       read_without].
  rewrite nth_error_map.
  destruct (nth_error params (read_position found)) as [x | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite may_take_renamed, var_free_b_equivariant, on_spine_renamed.
  destruct
    (may_take params metadata (read_variable found)
     && negb (var_free_b (read_variable found) handler)
     && on_spine (read_variable found) handler);
    [ | reflexivity ].
  rewrite extend_applications_equivariant.
  destruct
    (extend_applications known k (read_position found)
       (read_primitive found) body)
    as [body' | ]; cbn [option_map]; [ | reflexivity ].
  rewrite facts_under_params_equivariant.
  change (Prim (Unary (read_primitive found)) [Var (rename_variable r x)])
    with (rename_named r (Prim (Unary (read_primitive found)) [Var x])).
  rewrite facts_under_let_equivariant, extend_applications_equivariant.
  destruct
    (extend_applications
       (facts_under_let (facts_under_params known params)
          (read_variable found)
          (Prim (Unary (read_primitive found)) [Var x]))
       k (read_position found) (read_primitive found) (read_without found))
    as [without' | ]; cbn [option_map rename_step];
    [ rewrite map_app; reflexivity | reflexivity ].
Qed.

#[local] Hint Rewrite on_spine_renamed unbox_loop_read_equivariant :
  equivariance.

Lemma loop_of_renamed (r : renaming) (k : continuation) (state : loop_state) :
  rename_expr r (loop_of k state)
  = loop_of (rename_continuation r k) (rename_step r state).
Proof. equivariance. Qed.

Lemma unbox_loop_param_equivariant (r : renaming) (fuel : nat) :
  ∀ (known : facts) (k : continuation) (i : nat) (state : loop_state),
    unbox_loop_param fuel (rename_facts r known) (rename_continuation r k) i
      (rename_step r state)
    = rename_step r (unbox_loop_param fuel known k i state).
Proof.
  induction fuel as [ | fuel IH]; intros known k i state; [ reflexivity | ].
  destruct state as [[[params metadata] handler] body].
  cbn [unbox_loop_param rename_step].
  rewrite nth_error_map.
  destruct (nth_error params i) as [param | ]; cbn [option_map];
    [ | reflexivity ].
  pose proof (next_read_equivariant r [param] handler) as Hnext.
  cbn [map] in Hnext. rewrite Hnext.
  destruct (next_read [param] handler) as [found | ]; cbn [option_map];
    [ | reflexivity ].
  change
    (Read (read_variable (rename_read r found))
       (read_kind (rename_read r found))
       (read_primitive (rename_read r found)) i
       (read_without (rename_read r found)))
    with
      (rename_read r
         (Read (read_variable found) (read_kind found)
            (read_primitive found) i (read_without found))).
  rewrite unbox_loop_read_equivariant.
  destruct
    (unbox_loop_read known k params metadata handler body
       (Read (read_variable found) (read_kind found) (read_primitive found)
          i (read_without found)))
    as [state' | ]; cbn [option_map]; [ apply IH | reflexivity ].
Qed.

#[local] Hint Rewrite loop_of_renamed unbox_loop_param_equivariant :
  equivariance.

Lemma reads_var_named_renamed (r : renaming) (x : variable) (d : named) :
  reads_var_named (rename_variable r x) (rename_named r d)
  = reads_var_named x d.
Proof. equivariance. Qed.

#[local] Hint Rewrite reads_var_named_renamed : equivariance.

Lemma reads_var_renamed (r : renaming) (x : variable) (e : expr) :
  reads_var (rename_variable r x) (rename_expr r e) = reads_var x e.
Proof. equivariance. Qed.

#[local] Hint Rewrite reads_var_renamed : equivariance.

Lemma reads_left_renamed (r : renaming) (i : nat) (state : loop_state) :
  reads_left i (rename_step r state) = reads_left i state.
Proof. equivariance. Qed.

#[local] Hint Rewrite reads_left_renamed : equivariance.

Lemma unbox_loop_param_all_equivariant (r : renaming) (fuel : nat)
    (known : facts) (k : continuation) (i : nat) (state : loop_state) :
  unbox_loop_param_all fuel (rename_facts r known) (rename_continuation r k)
    i (rename_step r state)
  = rename_step r (unbox_loop_param_all fuel known k i state).
Proof. equivariance. Qed.

#[local] Hint Rewrite unbox_loop_param_all_equivariant : equivariance.

Lemma unbox_loop_params_equivariant (r : renaming) (fuel : nat)
    (known : facts) (k : continuation) (positions : list nat) :
  ∀ state : loop_state,
    unbox_loop_params fuel (rename_facts r known) (rename_continuation r k)
      positions (rename_step r state)
    = rename_step r (unbox_loop_params fuel known k positions state).
Proof. equivariance. Qed.

Lemma unbox_loop_reads_equivariant (r : renaming) (known : facts)
    (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  unbox_loop_reads (rename_facts r known) (rename_continuation r k)
    (map (rename_variable r) params) metadata (rename_expr r handler)
    (rename_expr r body)
  = rename_expr r (unbox_loop_reads known k params metadata handler body).
Proof.
  unfold unbox_loop_reads.
  rewrite lets_in_renamed, length_map, loop_of_renamed,
    <- unbox_loop_params_equivariant.
  reflexivity.
Qed.

#[local] Hint Rewrite unbox_loop_params_equivariant
  unbox_loop_reads_equivariant : equivariance.

Theorem unbox_loops_knowing_equivariant :
  unbox_loops_knowing_equivariant_statement.
Proof. equivariance. Qed.

End Renamed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

(* What is known holds variables that are free in the term or bound
   around the subterm. *)
Definition reads_within (names : list variable) (known : computed)
    : Prop :=
  ∀ entry : variable * (prim * list simple),
    entry ∈ known -> fst entry ∈ names.

Definition facts_within (names : list variable) (known : facts) : Prop :=
  blocks_within names (facts_blocks known)
  ∧ numbers_within names (facts_numbers known)
  ∧ reads_within names (facts_reads known).

Lemma forget_computed_within (names : list variable) (x : variable)
    (known : computed) :
  reads_within names known -> reads_within names (forget_computed x known).
Proof.
  intros Hreads entry Hin. unfold forget_computed in Hin.
  apply filter_In in Hin. exact (Hreads entry (proj1 Hin)).
Qed.

Lemma forget_computed_vars_within (names : list variable)
    (xs : list variable) :
  ∀ known : computed,
    reads_within names known ->
    reads_within names (forget_computed_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known Hreads; simpl;
    [ exact Hreads | apply IH, forget_computed_within, Hreads ].
Qed.

Lemma reads_under_let_within (names : list variable) (known : computed)
    (x : variable) (d : named) :
  reads_within names known ->
  reads_within (x :: names ++ free_variables_of_named d)
    (reads_under_let known x d).
Proof.
  intros Hreads.
  assert (Hforgot :
    reads_within (x :: names ++ free_variables_of_named d)
      (forget_computed x known)).
  { intros entry Hin. right. apply in_or_app. left.
    exact (forget_computed_within names x known Hreads entry Hin). }
  unfold reads_under_let.
  destruct d as [s | p args | | | | | ]; try exact Hforgot.
  destruct p as [p | | | | ]; try exact Hforgot.
  destruct args as [ | [y | sym | c] [ | arg args]]; try exact Hforgot.
  destruct (reads_a_box p && negb (Pos.eqb x y)); [ | exact Hforgot ].
  intros entry [<- | Hin]; [ left; reflexivity | exact (Hforgot entry Hin) ].
Qed.

Lemma facts_under_let_within (names : list variable) (known : facts)
    (x : variable) (d : named) :
  facts_within names known ->
  facts_within (x :: names ++ free_variables_of_named d)
    (facts_under_let known x d).
Proof.
  intros (Hblocks & Hnumbers & Hreads).
  split; [ apply known_under_let_within, Hblocks | ].
  split;
    [ apply numbers_under_let_within, Hnumbers
    | apply reads_under_let_within, Hreads ].
Qed.

Lemma facts_under_statics_within (names : list variable) (known : facts)
    (bindings : list static_binding) :
  facts_within names known ->
  facts_within (names ++ flat_map free_variables_of_binding bindings)
    (facts_under_statics known bindings).
Proof.
  intros (Hblocks & Hnumbers & Hreads).
  split; [ apply known_under_statics_within, Hblocks | split ].
  - intros entry Hentry.
    eapply incl_tran; [ exact (Hnumbers entry Hentry) | ].
    apply incl_appl, incl_refl.
  - intros entry Hentry. apply in_or_app. left. exact (Hreads entry Hentry).
Qed.

Lemma facts_under_params_within (names : list variable) (known : facts)
    (params : list variable) :
  facts_within names known ->
  facts_within names (facts_under_params known params).
Proof.
  intros (Hblocks & Hnumbers & Hreads).
  split; [ apply forget_vars_within, Hblocks | ].
  split;
    [ apply forget_wrapped_vars_within, Hnumbers
    | apply forget_computed_vars_within, Hreads ].
Qed.

(* What an application is given is of the argument or of what is
   known. *)
Lemma projection_of_free (names : list variable) (known : facts)
    (p : unary_primitive) (arg projection : simple) :
  facts_within names known ->
  projection_of known p arg = Some projection ->
  incl (free_variables_of_simple projection)
    (free_variables_of_simple arg ++ names).
Proof.
  intros (Hblocks & Hnumbers & Hreads). unfold projection_of. cbv zeta.
  destruct (reads_a_box p); [ | discriminate ].
  remember (Prim (Unary p) [arg]) as d eqn:Hd.
  assert (Hfree : free_variables_of_named d = free_variables_of_simple arg).
  { subst d. cbn [free_variables_of_named]. unfold free_variables_of_simples.
    cbn [flat_map]. apply app_nil_r. }
  pose proof (fold_named_free d) as Hfold.
  pose proof (load_known_named_free names (facts_blocks known) d Hblocks)
    as Hload.
  pose proof (unbox_known_named_free names (facts_numbers known) d Hnumbers)
    as Hunbox.
  rewrite Hfree in Hfold, Hload, Hunbox. clear Hd Hfree.
  destruct (fold_named d);
    [ intros [= <-]; apply incl_appl; exact Hfold | .. ];
    (destruct (load_known_named (facts_blocks known) d);
     [ intros [= <-]; exact Hload | .. ];
     (destruct (unbox_known_named (facts_numbers known) d);
      [ intros [= <-]; exact Hunbox | .. ];
      (destruct (computed_by (facts_reads known) (Unary p) [arg])
         as [x | ] eqn:Hby;
       [ intros [= <-] z [<- | []]; apply in_or_app; right;
         exact (Hreads _ (computed_by_sound _ _ _ _ Hby))
       | discriminate ]))).
Qed.

Lemma extend_apply_cont_free (names : list variable) (known : facts)
    (k : continuation) (i : nat) (p : unary_primitive) (ac ac' : apply_cont) :
  facts_within names known ->
  extend_apply_cont known k i p ac = Some ac' ->
  incl (free_variables_of_apply_cont ac')
    (free_variables_of_apply_cont ac ++ names)
  ∧ apply_cont_continuations ac' = apply_cont_continuations ac.
Proof.
  intros Hknown. unfold extend_apply_cont.
  destruct (Pos.eqb (ac_continuation ac) k).
  - unfold extend_args.
    destruct (nth_error (ac_args ac) i) as [arg | ] eqn:Hnth;
      [ | discriminate ].
    destruct (projection_of known p arg) as [projection | ] eqn:Hprojection;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    unfold free_variables_of_apply_cont, free_variables_of_simples.
    cbn [ac_args]. rewrite flat_map_app. cbn [flat_map]. rewrite app_nil_r.
    intros y Hy. apply in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ apply in_or_app; left; exact Hy | ].
    apply (projection_of_free names known p arg projection Hknown
             Hprojection)
      in Hy.
    apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
    apply in_flat_map. exists arg.
    split; [ exact (nth_error_In _ _ Hnth) | exact Hy ].
  - intros [= <-]. split; [ apply incl_appl, incl_refl | reflexivity ].
Qed.

Lemma extend_arms_free (names : list variable) (known : facts)
    (k : continuation) (i : nat) (p : unary_primitive) :
  ∀ arms arms' : list (Z * apply_cont),
    facts_within names known ->
    extend_arms known k i p arms = Some arms' ->
    incl (flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms')
      (flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms
       ++ names)
    ∧ flat_map (fun arm => apply_cont_continuations (snd arm)) arms'
      = flat_map (fun arm => apply_cont_continuations (snd arm)) arms.
Proof.
  induction arms as [ | [discriminant ac] arms IH]; intros arms' Hknown;
    cbn [extend_arms flat_map snd].
  - intros [= <-]. split; [ intros y [] | reflexivity ].
  - destruct (extend_apply_cont known k i p ac) as [ac' | ] eqn:Hac;
      [ | discriminate ].
    destruct (extend_arms known k i p arms) as [rest | ]; [ | discriminate ].
    intros [= <-].
    destruct (extend_apply_cont_free names known k i p ac ac' Hknown Hac)
      as [Hv Hk].
    destruct (IH rest Hknown eq_refl) as [Hv' Hk'].
    cbn [flat_map snd]. rewrite Hk, Hk'. split; [ | reflexivity ].
    intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy].
    + apply Hv, in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
    + apply Hv', in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
Qed.

Fixpoint extend_applications_scoped (e : expr) :
  ∀ (names : list variable) (known : facts) (k : continuation) (i : nat)
    (p : unary_primitive) (e' : expr),
    facts_within names known ->
    extend_applications known k i p e = Some e' ->
    scoped_under names e e'
with extend_in_handler_scoped (h : continuation_handler) :
  ∀ (names : list variable) (known : facts) (k : continuation) (i : nat)
    (p : unary_primitive) (h' : continuation_handler),
    facts_within names known ->
    extend_in_handler known k i p h = Some h' ->
    handler_code_closed h = true ->
    handler_code_closed h' = true
    ∧ incl (free_variables_of_handler h') (free_variables_of_handler h ++ names)
    ∧ incl (free_continuations_of_handler h')
        (free_continuations_of_handler h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k0 h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros names known k i p e' Hknown; cbn [extend_applications].
    + destruct (extend_applications (facts_under_let known x d) k i p body)
        as [body' | ] eqn:Hbody; [ | discriminate ].
      intros [= <-].
      apply scoped_under_let; [ apply incl_appl, incl_refl | ].
      exact (extend_applications_scoped body _ _ k i p body'
        (facts_under_let_within names known x d Hknown) Hbody).
    + destruct
        (extend_applications (facts_under_statics known bindings) k i p body)
        as [body' | ] eqn:Hbody; [ | discriminate ].
      intros [= <-]. apply scoped_under_let_static.
      exact (extend_applications_scoped body _ _ k i p body'
        (facts_under_statics_within names known bindings Hknown) Hbody).
    + destruct h as [params metadata handler]. cbn [extend_in_handler].
      destruct
        (extend_applications (facts_under_params known params) k i p handler)
        as [handler' | ] eqn:Hhandler; [ | discriminate ].
      pose proof (extend_applications_scoped handler names _ k i p handler'
        (facts_under_params_within names known params Hknown) Hhandler)
        as Hin_handler.
      destruct (Pos.eqb k0 k).
      * intros [= <-].
        apply scoped_under_let_cont;
          [ exact Hin_handler | apply scoped_under_refl ].
      * destruct (extend_applications known k i p body) as [body' | ]
          eqn:Hbody; [ | discriminate ].
        intros [= <-].
        apply scoped_under_let_cont; [ exact Hin_handler | ].
        exact (extend_applications_scoped body names known k i p body' Hknown
          Hbody).
    + destruct (is_bound k handlers);
        [ intros [= <-]; apply scoped_under_refl | ].
      destruct
        (all_some
           (map
              (fun '(k', h) =>
                 let* h := extend_in_handler known k i p h in
                 Some (k', h))
              handlers))
        as [handlers' | ] eqn:Hhandlers; [ | discriminate ].
      destruct (extend_applications known k i p body) as [body' | ]
        eqn:Hbody; [ | discriminate ].
      intros [= <-] Hclosed. scope_in Hclosed. scope.
      apply andb_true_iff in Hclosed. destruct Hclosed as [Hgroup Hclosed].
      destruct
        (extend_applications_scoped body names known k i p body' Hknown Hbody
           Hclosed)
        as (Hclosed' & Hv & Hk).
      assert (Hin_handlers :
        forallb (fun '(_, h) => handler_code_closed h) handlers' = true
        ∧ incl
            (flat_map (fun '(_, h) => free_variables_of_handler h) handlers')
            (flat_map (fun '(_, h) => free_variables_of_handler h) handlers
             ++ names)
        ∧ incl
            (flat_map (fun '(_, h) => free_continuations_of_handler h)
               handlers')
            (flat_map (fun '(_, h) => free_continuations_of_handler h)
               handlers)
        ∧ map fst handlers' = map fst handlers).
      { clear Hbody Hclosed Hclosed' Hv Hk. revert handlers' Hhandlers Hgroup.
        induction handlers as [ | [k1 h1] handlers IH]; intros handlers';
          simpl.
        - intros [= <-] _.
          split; [ reflexivity | ].
          split; [ intros y [] | split; [ intros y [] | reflexivity ] ].
        - destruct (extend_in_handler known k i p h1) as [h1' | ] eqn:Hh1;
            [ | discriminate ].
          destruct
            (all_some
               (map
                  (fun '(k', h) =>
                     let* h := extend_in_handler known k i p h in
                     Some (k', h))
                  handlers))
            as [rest | ]; [ | discriminate ].
          intros [= <-] Hgroup. apply andb_true_iff in Hgroup.
          destruct Hgroup as [Hfirst Hrest].
          destruct
            (extend_in_handler_scoped h1 names known k i p h1' Hknown Hh1
               Hfirst)
            as (Hfirst' & Hv1 & Hk1).
          destruct (IH rest eq_refl Hrest) as (Hrest' & Hv2 & Hk2 & Hnames).
          simpl. rewrite Hfirst', Hrest', Hnames.
          split; [ reflexivity | ].
          split; [ | split; [ apply incl_app_app; assumption | reflexivity ] ].
          intros y Hy. apply in_app_or in Hy. apply in_or_app.
          destruct Hy as [Hy | Hy].
          + apply Hv1, in_app_or in Hy.
            destruct Hy as [Hy | Hy];
              [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
          + apply Hv2, in_app_or in Hy.
            destruct Hy as [Hy | Hy];
              [ left; apply in_or_app; right; exact Hy | right; exact Hy ]. }
      destruct Hin_handlers as (Hgroup' & Hgroup_v & Hgroup_k & Hnames).
      rewrite Hgroup', Hclosed', Hnames. split; [ reflexivity | split ].
      * intros y Hy. apply in_app_or in Hy. apply in_or_app.
        destruct Hy as [Hy | Hy].
        -- apply Hgroup_v, in_app_or in Hy.
           destruct Hy as [Hy | Hy];
             [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
        -- apply Hv, in_app_or in Hy.
           destruct Hy as [Hy | Hy];
             [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
      * apply incl_without, incl_app_app; assumption.
    + destruct
        (Pos.eqb (ap_continuation ap) k || Pos.eqb (ap_exn_continuation ap) k);
        [ discriminate | intros [= <-]; apply scoped_under_refl ].
    + destruct (extend_apply_cont known k i p ac) as [ac' | ] eqn:Hac;
        [ | discriminate ].
      intros [= <-] _. scope.
      destruct (extend_apply_cont_free names known k i p ac ac' Hknown Hac)
        as [Hv Hk].
      split; [ reflexivity | split; [ exact Hv | ] ].
      rewrite Hk. apply incl_refl.
    + destruct (extend_arms known k i p arms) as [arms' | ] eqn:Harms;
        [ | discriminate ].
      intros [= <-] _. scope.
      destruct (extend_arms_free names known k i p arms arms' Hknown Harms)
        as [Hv Hk].
      split; [ reflexivity | split; [ | rewrite Hk; apply incl_refl ] ].
      intros y Hy. apply in_app_or in Hy. apply in_or_app.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; left; exact Hy | ].
      apply Hv, in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
    + intros [= <-]. apply scoped_under_refl.
  - destruct h as [params metadata handler].
    intros names known k i p h' Hknown. cbn [extend_in_handler].
    destruct
      (extend_applications (facts_under_params known params) k i p handler)
      as [handler' | ] eqn:Hhandler; [ | discriminate ].
    intros [= <-] Hclosed. scope_in Hclosed. scope.
    destruct
      (extend_applications_scoped handler names _ k i p handler'
         (facts_under_params_within names known params Hknown) Hhandler
         Hclosed)
      as (Hclosed' & Hv & Hk).
    split; [ exact Hclosed' | split; [ | exact Hk ] ].
    intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hparams].
    apply Hv, in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
    apply in_without. split; assumption.
Qed.

Lemma scoped_under_trans (names : list variable) (e e' e'' : expr) :
  scoped_under names e e' ->
  scoped_under names e' e'' ->
  scoped_under names e e''.
Proof.
  intros Hfirst Hsecond Hclosed.
  destruct (Hfirst Hclosed) as (Hclosed' & Hv & Hk).
  destruct (Hsecond Hclosed') as (Hclosed'' & Hv' & Hk').
  split; [ exact Hclosed'' | split; [ | eapply incl_tran; eassumption ] ].
  intros y Hy. apply Hv', in_app_or in Hy.
  destruct Hy as [Hy | Hy]; [ exact (Hv y Hy) | ].
  apply in_or_app. right. exact Hy.
Qed.

(* The [Let] at the head of the handler made a parameter. *)
Lemma unbox_head_scoped (names : list variable) (k : continuation)
    (params : list variable) (metadata metadata' : handler_metadata)
    (f : variable) (kind : flambda_kind) (d : named) (rest body body' : expr) :
  scoped_under names body body' ->
  scoped_under names
    (Let_cont_non_recursive k (Handler params metadata (Let f kind d rest))
       body)
    (Let_cont_non_recursive k (Handler (params ++ [f]) metadata' rest) body').
Proof.
  intros Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hrest Hclosed].
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  rewrite Hrest, Hclosed'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy].
    + left. apply in_or_app. left.
      apply in_without in Hy. destruct Hy as [Hy Hparams].
      apply in_without. split.
      * apply in_or_app. right. apply in_without.
        split; [ exact Hy | ].
        intros [Heq | []]. apply Hparams, in_or_app. right. left. exact Heq.
      * intros Hin. apply Hparams, in_or_app. left. exact Hin.
    + apply Hv, in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
  - apply incl_app_app; [ apply incl_refl | apply incl_without, Hk ].
Qed.

Lemma unbox_heads_scoped (handler : expr) :
  ∀ (names : list variable) (known : facts) (k : continuation)
    (params : list variable) (metadata : handler_metadata) (body : expr),
    facts_within names known ->
    scoped_under names
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (unbox_heads known k params metadata handler body).
Proof.
  induction handler
    as [ f kind d rest IH | bindings rest _ | k0 h rest _ | handlers rest _
       | ap | ac | scrutinee arms | ];
    intros names known k params metadata body Hknown;
    try (simpl; apply scoped_under_refl).
  destruct d
    as [ s | pr args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure ];
    try (simpl; apply scoped_under_refl).
  destruct pr as [p | b | v | hp | rp]; try (simpl; apply scoped_under_refl).
  destruct args as [ | arg args]; try (simpl; apply scoped_under_refl).
  destruct arg as [x | sym | c]; try (simpl; apply scoped_under_refl).
  destruct args as [ | arg' args]; try (simpl; apply scoped_under_refl).
  simpl.
  destruct (may_take params metadata f); [ | apply scoped_under_refl ].
  destruct (position x params) as [i | ]; [ | apply scoped_under_refl ].
  destruct (extend_applications known k i p body) as [body' | ] eqn:Hbody;
    [ | apply scoped_under_refl ].
  eapply scoped_under_trans; [ | apply IH; exact Hknown ].
  apply unbox_head_scoped.
  exact (extend_applications_scoped body names known k i p body' Hknown
    Hbody).
Qed.

Fixpoint unbox_params_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : facts),
    facts_within names known ->
    scoped_under names e (unbox_params_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown; cbn [unbox_params_knowing];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply incl_appl, incl_refl | ].
    apply unbox_params_knowing_scoped, facts_under_let_within, Hknown.
  - apply scoped_under_let_static.
    apply unbox_params_knowing_scoped, facts_under_statics_within, Hknown.
  - eapply scoped_under_trans; [ | apply unbox_heads_scoped, Hknown ].
    apply scoped_under_let_cont; apply unbox_params_knowing_scoped;
      [ apply facts_under_params_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply unbox_params_knowing_scoped, Hknown.
Qed.

Theorem unbox_known_params_keeps_scope :
  unbox_known_params_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e.
  apply scoped_under_none, unbox_params_knowing_scoped.
  split; [ split; intros entry [] | split; intros entry [] ].
Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Section By_scoped.

Variable extend : facts -> list simple -> option (list simple).

(* What an application is given is of its arguments or of what is
   known. *)
Hypothesis extend_free :
  ∀ (names : list variable) (known : facts) (args args' : list simple),
    facts_within names known ->
    extend known args = Some args' ->
    incl (free_variables_of_simples args')
      (free_variables_of_simples args ++ names).

Lemma extend_apply_cont_by_free (names : list variable) (known : facts)
    (k : continuation) (ac ac' : apply_cont) :
  facts_within names known ->
  extend_apply_cont_by extend known k ac = Some ac' ->
  incl (free_variables_of_apply_cont ac')
    (free_variables_of_apply_cont ac ++ names)
  ∧ apply_cont_continuations ac' = apply_cont_continuations ac.
Proof.
  intros Hknown. unfold extend_apply_cont_by.
  destruct (Pos.eqb (ac_continuation ac) k).
  - destruct (extend known (ac_args ac)) as [args' | ] eqn:Hextend;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (extend_free names known _ _ Hknown Hextend).
  - intros [= <-]. split; [ apply incl_appl, incl_refl | reflexivity ].
Qed.

Lemma extend_arms_by_free (names : list variable) (known : facts)
    (k : continuation) :
  ∀ arms arms' : list (Z * apply_cont),
    facts_within names known ->
    extend_arms_by extend known k arms = Some arms' ->
    incl (flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms')
      (flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms
       ++ names)
    ∧ flat_map (fun arm => apply_cont_continuations (snd arm)) arms'
      = flat_map (fun arm => apply_cont_continuations (snd arm)) arms.
Proof.
  induction arms as [ | [discriminant ac] arms IH]; intros arms' Hknown;
    cbn [extend_arms_by flat_map snd].
  - intros [= <-]. split; [ intros y [] | reflexivity ].
  - destruct (extend_apply_cont_by extend known k ac) as [ac' | ] eqn:Hac;
      [ | discriminate ].
    destruct (extend_arms_by extend known k arms) as [rest | ];
      [ | discriminate ].
    intros [= <-].
    destruct (extend_apply_cont_by_free names known k ac ac' Hknown Hac)
      as [Hv Hk].
    destruct (IH rest Hknown eq_refl) as [Hv' Hk'].
    cbn [flat_map snd]. rewrite Hk, Hk'. split; [ | reflexivity ].
    intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy].
    + apply Hv, in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
    + apply Hv', in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
Qed.

Fixpoint extend_applications_by_scoped (e : expr) :
  ∀ (names : list variable) (known : facts) (k : continuation) (e' : expr),
    facts_within names known ->
    extend_applications_by extend known k e = Some e' ->
    scoped_under names e e'
with extend_in_handler_by_scoped (h : continuation_handler) :
  ∀ (names : list variable) (known : facts) (k : continuation)
    (h' : continuation_handler),
    facts_within names known ->
    extend_in_handler_by extend known k h = Some h' ->
    handler_code_closed h = true ->
    handler_code_closed h' = true
    ∧ incl (free_variables_of_handler h') (free_variables_of_handler h ++ names)
    ∧ incl (free_continuations_of_handler h')
        (free_continuations_of_handler h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k0 h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros names known k e' Hknown; cbn [extend_applications_by].
    + destruct
        (extend_applications_by extend (facts_under_let known x d) k body)
        as [body' | ] eqn:Hbody; [ | discriminate ].
      intros [= <-].
      apply scoped_under_let; [ apply incl_appl, incl_refl | ].
      exact (extend_applications_by_scoped body _ _ k body'
        (facts_under_let_within names known x d Hknown) Hbody).
    + destruct
        (extend_applications_by extend (facts_under_statics known bindings) k
           body)
        as [body' | ] eqn:Hbody; [ | discriminate ].
      intros [= <-]. apply scoped_under_let_static.
      exact (extend_applications_by_scoped body _ _ k body'
        (facts_under_statics_within names known bindings Hknown) Hbody).
    + destruct h as [params metadata handler]. cbn [extend_in_handler_by].
      destruct
        (extend_applications_by extend (facts_under_params known params) k
           handler)
        as [handler' | ] eqn:Hhandler; [ | discriminate ].
      pose proof (extend_applications_by_scoped handler names _ k handler'
        (facts_under_params_within names known params Hknown) Hhandler)
        as Hin_handler.
      destruct (Pos.eqb k0 k).
      * intros [= <-].
        apply scoped_under_let_cont;
          [ exact Hin_handler | apply scoped_under_refl ].
      * destruct (extend_applications_by extend known k body) as [body' | ]
          eqn:Hbody; [ | discriminate ].
        intros [= <-].
        apply scoped_under_let_cont; [ exact Hin_handler | ].
        exact (extend_applications_by_scoped body names known k body' Hknown
          Hbody).
    + destruct (is_bound k handlers);
        [ intros [= <-]; apply scoped_under_refl | ].
      destruct
        (all_some
           (map
              (fun '(k', h) =>
                 let* h := extend_in_handler_by extend known k h in
                 Some (k', h))
              handlers))
        as [handlers' | ] eqn:Hhandlers; [ | discriminate ].
      destruct (extend_applications_by extend known k body) as [body' | ]
        eqn:Hbody; [ | discriminate ].
      intros [= <-] Hclosed. scope_in Hclosed. scope.
      apply andb_true_iff in Hclosed. destruct Hclosed as [Hgroup Hclosed].
      destruct
        (extend_applications_by_scoped body names known k body' Hknown Hbody
           Hclosed)
        as (Hclosed' & Hv & Hk).
      assert (Hin_handlers :
        forallb (fun '(_, h) => handler_code_closed h) handlers' = true
        ∧ incl
            (flat_map (fun '(_, h) => free_variables_of_handler h) handlers')
            (flat_map (fun '(_, h) => free_variables_of_handler h) handlers
             ++ names)
        ∧ incl
            (flat_map (fun '(_, h) => free_continuations_of_handler h)
               handlers')
            (flat_map (fun '(_, h) => free_continuations_of_handler h)
               handlers)
        ∧ map fst handlers' = map fst handlers).
      { clear Hbody Hclosed Hclosed' Hv Hk. revert handlers' Hhandlers Hgroup.
        induction handlers as [ | [k1 h1] handlers IH]; intros handlers';
          simpl.
        - intros [= <-] _.
          split; [ reflexivity | ].
          split; [ intros y [] | split; [ intros y [] | reflexivity ] ].
        - destruct (extend_in_handler_by extend known k h1) as [h1' | ]
            eqn:Hh1; [ | discriminate ].
          destruct
            (all_some
               (map
                  (fun '(k', h) =>
                     let* h := extend_in_handler_by extend known k h in
                     Some (k', h))
                  handlers))
            as [rest | ]; [ | discriminate ].
          intros [= <-] Hgroup. apply andb_true_iff in Hgroup.
          destruct Hgroup as [Hfirst Hrest].
          destruct
            (extend_in_handler_by_scoped h1 names known k h1' Hknown Hh1
               Hfirst)
            as (Hfirst' & Hv1 & Hk1).
          destruct (IH rest eq_refl Hrest) as (Hrest' & Hv2 & Hk2 & Hnames).
          simpl. rewrite Hfirst', Hrest', Hnames.
          split; [ reflexivity | ].
          split; [ | split; [ apply incl_app_app; assumption | reflexivity ] ].
          intros y Hy. apply in_app_or in Hy. apply in_or_app.
          destruct Hy as [Hy | Hy].
          + apply Hv1, in_app_or in Hy.
            destruct Hy as [Hy | Hy];
              [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
          + apply Hv2, in_app_or in Hy.
            destruct Hy as [Hy | Hy];
              [ left; apply in_or_app; right; exact Hy | right; exact Hy ]. }
      destruct Hin_handlers as (Hgroup' & Hgroup_v & Hgroup_k & Hnames).
      rewrite Hgroup', Hclosed', Hnames. split; [ reflexivity | split ].
      * intros y Hy. apply in_app_or in Hy. apply in_or_app.
        destruct Hy as [Hy | Hy].
        -- apply Hgroup_v, in_app_or in Hy.
           destruct Hy as [Hy | Hy];
             [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
        -- apply Hv, in_app_or in Hy.
           destruct Hy as [Hy | Hy];
             [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
      * apply incl_without, incl_app_app; assumption.
    + destruct
        (Pos.eqb (ap_continuation ap) k || Pos.eqb (ap_exn_continuation ap) k);
        [ discriminate | intros [= <-]; apply scoped_under_refl ].
    + destruct (extend_apply_cont_by extend known k ac) as [ac' | ] eqn:Hac;
        [ | discriminate ].
      intros [= <-] _. scope.
      destruct (extend_apply_cont_by_free names known k ac ac' Hknown Hac)
        as [Hv Hk].
      split; [ reflexivity | split; [ exact Hv | ] ].
      rewrite Hk. apply incl_refl.
    + destruct (extend_arms_by extend known k arms) as [arms' | ] eqn:Harms;
        [ | discriminate ].
      intros [= <-] _. scope.
      destruct (extend_arms_by_free names known k arms arms' Hknown Harms)
        as [Hv Hk].
      split; [ reflexivity | split; [ | rewrite Hk; apply incl_refl ] ].
      intros y Hy. apply in_app_or in Hy. apply in_or_app.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; left; exact Hy | ].
      apply Hv, in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
    + intros [= <-]. apply scoped_under_refl.
  - destruct h as [params metadata handler].
    intros names known k h' Hknown. cbn [extend_in_handler_by].
    destruct
      (extend_applications_by extend (facts_under_params known params) k
         handler)
      as [handler' | ] eqn:Hhandler; [ | discriminate ].
    intros [= <-] Hclosed. scope_in Hclosed. scope.
    destruct
      (extend_applications_by_scoped handler names _ k handler'
         (facts_under_params_within names known params Hknown) Hhandler
         Hclosed)
      as (Hclosed' & Hv & Hk).
    split; [ exact Hclosed' | split; [ | exact Hk ] ].
    intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hparams].
    apply Hv, in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
    apply in_without. split; assumption.
Qed.

End By_scoped.

(* What the two functions give an application. *)
Lemma extend_args_free (i : nat) (p : unary_primitive)
    (names : list variable) (known : facts) (args args' : list simple) :
  facts_within names known ->
  extend_args known i p args = Some args' ->
  incl (free_variables_of_simples args')
    (free_variables_of_simples args ++ names).
Proof.
  intros Hknown. unfold extend_args.
  destruct (nth_error args i) as [arg | ] eqn:Hnth; [ | discriminate ].
  simpl.
  destruct (projection_of known p arg) as [projection | ] eqn:Hprojection;
    [ | discriminate ].
  intros [= <-]. unfold free_variables_of_simples.
  rewrite flat_map_app. cbn [flat_map]. rewrite app_nil_r.
  intros y Hy. apply in_app_or in Hy.
  destruct Hy as [Hy | Hy]; [ apply in_or_app; left; exact Hy | ].
  apply (projection_of_free names known p arg projection Hknown Hprojection)
    in Hy.
  apply in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
  apply in_flat_map. exists arg.
  split; [ exact (nth_error_In _ _ Hnth) | exact Hy ].
Qed.

Lemma made_field_free (names : list variable) (known : facts)
    (p : unary_primitive) (kind : flambda_kind) (arg projection : simple) :
  facts_within names known ->
  made_field known p kind arg = Some projection ->
  incl (free_variables_of_simple projection) names.
Proof.
  intros [[Hvars _] _] Hmade. unfold made_field in Hmade.
  destruct p as [access mut field | | | | | | | | | | | | | | ];
    try discriminate Hmade.
  assert (Hfield :
    match arg with
    | Var y =>
      let* block := lookup y (known_vars (facts_blocks known)) in
      let* contents := field_loaded access field block in
      if kind_eqb kind Value then Some contents else None
    | _ => None
    end
    = Some projection)
    by (destruct mut; [ discriminate Hmade | exact Hmade .. ]).
  clear Hmade. destruct arg as [y | sym | c]; try discriminate Hfield.
  destruct (lookup y (known_vars (facts_blocks known))) as [block | ]
    eqn:Hblock; [ | discriminate ].
  simpl in Hfield.
  destruct (field_loaded access field block) as [contents | ] eqn:Hloaded;
    [ | discriminate ].
  simpl in Hfield.
  destruct (kind_eqb kind Value); [ | discriminate ].
  injection Hfield as <-.
  destruct (field_loaded_spec _ _ _ _ Hloaded)
    as (tag & size & _ & _ & _ & _ & Hnth).
  exact (Hvars (y, block) (lookup_in _ _ _ Hblock) contents
    (nth_error_In _ _ Hnth)).
Qed.

Lemma made_number_free (names : list variable) (known : facts)
    (p : unary_primitive) (kind : flambda_kind) (arg projection : simple) :
  facts_within names known ->
  made_number known p kind arg = Some projection ->
  incl (free_variables_of_simple projection) names.
Proof.
  intros (_ & Hnumbers & _) Hmade. unfold made_number in Hmade.
  destruct p as [ | | | | | | | | | | boxed | | | | ]; try discriminate Hmade;
    (destruct arg as [y | sym | c]; try discriminate Hmade).
  - destruct (unwrapped (facts_numbers known) (Boxed boxed) (Var y))
      as [contents | ] eqn:Hunwrapped; [ | discriminate ].
    simpl in Hmade.
    destruct (kind_eqb kind (unboxed_kind boxed)); [ | discriminate ].
    injection Hmade as <-.
    exact (unwrapped_free names (facts_numbers known) _ _ _ Hnumbers
      Hunwrapped).
  - destruct (unwrapped (facts_numbers known) Tagged (Var y))
      as [contents | ] eqn:Hunwrapped; [ | discriminate ].
    simpl in Hmade.
    destruct (kind_eqb kind (Naked_number Kind_naked_immediate));
      [ | discriminate ].
    injection Hmade as <-.
    exact (unwrapped_free names (facts_numbers known) _ _ _ Hnumbers
      Hunwrapped).
Qed.

Lemma extend_args_certain_free (variant : bool)
    (tests params : list variable)
    (handler without : expr) (i : nat) (p : unary_primitive)
    (kind : flambda_kind) (names : list variable) (known : facts)
    (args args' : list simple) :
  facts_within names known ->
  extend_args_certain variant tests params handler without known i p kind
    args
  = Some args' ->
  incl (free_variables_of_simples args')
    (free_variables_of_simples args ++ names).
Proof.
  intros Hknown. unfold extend_args_certain.
  destruct (nth_error args i) as [arg | ]; [ | discriminate ].
  simpl.
  assert (Hlast :
    ∀ last : simple,
      incl (free_variables_of_simple last) names ->
      incl (free_variables_of_simples (args ++ [last]))
        (free_variables_of_simples args ++ names)).
  { intros last Hlast. unfold free_variables_of_simples.
    rewrite flat_map_app. cbn [flat_map]. rewrite app_nil_r.
    apply incl_app_app; [ apply incl_refl | exact Hlast ]. }
  destruct
    (never_reads (only_of tests (known_immediates params args)) handler
       without);
    [ intros [= <-]; apply Hlast; intros z [] | ].
  destruct (certain_projection known p kind arg) as [projection | ]
    eqn:Hcertain.
  - intros [= <-]. apply Hlast. unfold certain_projection in Hcertain.
    assert (Hmade :
      match made_field known p kind arg with
      | Some contents => Some contents
      | None => made_number known p kind arg
      end
      = Some projection ->
      incl (free_variables_of_simple projection) names).
    { destruct (made_field known p kind arg) as [contents | ] eqn:Hfield.
      - intros [= <-].
        exact (made_field_free names known p kind arg contents Hknown
          Hfield).
      - exact (made_number_free names known p kind arg projection Hknown). }
    destruct (projection_of known p arg) as [[y | sym | c] | ];
      try exact (Hmade Hcertain).
    destruct (kind_eqb (kind_of_const c) kind); [ | discriminate ].
    injection Hcertain as <-. intros z [].
  - destruct (no_value_on known p arg); [ | discriminate ].
    intros [= <-]. apply Hlast. intros z [].
Qed.

(* An expression without one of its [Let]s: its code is closed if the
   code of the expression is, its free variables are those of the
   expression and the variable of the [Let], and it has no more free
   continuations. *)
Lemma removes_scoped (f : variable) (kind : flambda_kind)
    (p : unary_primitive) (x : variable) (e e' : expr) :
  removes f kind p x e e' ->
  code_closed e = true ->
  code_closed e' = true
  ∧ incl (free_variables_of_expr e') (f :: free_variables_of_expr e)
  ∧ incl (free_continuations_of_expr e') (free_continuations_of_expr e).
Proof.
  intros Hremoves.
  induction Hremoves
    as [ rest | y kind0 d body body' _ _ _ IH
       | k params metadata handler handler' body _ _ _ IH
       | k h body body' _ IH ];
    intros Hclosed; scope_in Hclosed; scope.
  - split; [ exact Hclosed | split; [ | apply incl_refl ] ].
    intros y Hy. destruct (Pos.eq_dec y f) as [-> | Hne];
      [ left; reflexivity | right ].
    apply in_or_app. right. apply in_without.
    split; [ exact Hy | intros [Heq | []]; congruence ].
  - destruct (IH Hclosed) as (Hclosed' & Hv & Hk).
    split; [ exact Hclosed' | split; [ | exact Hk ] ].
    intros z Hz. apply in_app_or in Hz.
    destruct Hz as [Hz | Hz]; [ right; apply in_or_app; left; exact Hz | ].
    apply in_without in Hz. destruct Hz as [Hz Hy].
    destruct (Hv z Hz) as [Heq | Hin]; [ left; exact Heq | right ].
    apply in_or_app. right. apply in_without. split; assumption.
  - apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
    destruct (IH Hfirst) as (Hfirst' & Hv & Hk).
    rewrite Hfirst', Hrest. split; [ reflexivity | split ].
    + intros z Hz. apply in_app_or in Hz.
      destruct Hz as [Hz | Hz]; [ | right; apply in_or_app; right; exact Hz ].
      apply in_without in Hz. destruct Hz as [Hz Hparams].
      destruct (Hv z Hz) as [Heq | Hin]; [ left; exact Heq | right ].
      apply in_or_app. left. apply in_without. split; assumption.
    + apply incl_app_app; [ exact Hk | apply incl_refl ].
  - apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
    destruct (IH Hrest) as (Hrest' & Hv & Hk).
    rewrite Hfirst, Hrest'. split; [ reflexivity | split ].
    + intros z Hz. apply in_app_or in Hz.
      destruct Hz as [Hz | Hz]; [ right; apply in_or_app; left; exact Hz | ].
      destruct (Hv z Hz) as [Heq | Hin]; [ left; exact Heq | right ].
      apply in_or_app. right. exact Hin.
    + apply incl_app_app; [ apply incl_refl | apply incl_without, Hk ].
Qed.

(* What a primitive applied to [f] in the place of one of the variables
   [same] mentions, and what a [Let] of [f] mentions. *)
Lemma shared_named_free (p : unary_primitive) (x f : variable)
    (same : list variable) (d : named) :
  incl
    (free_variables_of_named
       (if reads_same p x (read_of_same f same d)
        then Simple (Var f)
        else read_of_same f same d))
    (f :: free_variables_of_named d).
Proof.
  destruct (reads_same p x (read_of_same f same d)).
  - intros z [<- | []]. left. reflexivity.
  - destruct d as [s | pr args | | | | | ]; try (apply incl_tl, incl_refl).
    destruct pr as [p' | b | v | hp | rp]; try (apply incl_tl, incl_refl).
    destruct args as [ | [z | sym | c] [ | arg args]];
      try (apply incl_tl, incl_refl).
    cbn [read_of_same].
    destruct (among z same); [ | apply incl_tl, incl_refl ].
    intros y [<- | []]. left. reflexivity.
Qed.

(* An expression with reads shared: its code is closed as the code of
   the expression is, its free variables are those of the expression
   and the variable of the read, and its free continuations are those
   of the expression. *)
Fixpoint share_reads_scoped (p : unary_primitive) (x f : variable)
    (e : expr) :
  ∀ same : list variable,
    code_closed (share_reads p x f same e) = code_closed e
    ∧ incl (free_variables_of_expr (share_reads p x f same e))
        (f :: free_variables_of_expr e)
    ∧ free_continuations_of_expr (share_reads p x f same e)
      = free_continuations_of_expr e.
Proof.
  destruct e
    as [ y kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros same; cbn [share_reads];
    try (split; [ reflexivity
                | split; [ apply incl_tl, incl_refl | reflexivity ] ]).
  - set (shared := reads_same p x (read_of_same f same d)).
    set (same' := if shared then y :: same else without [y] same).
    assert (Hbody :
      code_closed
        (if Pos.eqb y x || Pos.eqb y f
         then body
         else share_reads p x f same' body)
      = code_closed body
      ∧ incl
          (without [y]
             (free_variables_of_expr
                (if Pos.eqb y x || Pos.eqb y f
                 then body
                 else share_reads p x f same' body)))
          (f :: without [y] (free_variables_of_expr body))
      ∧ free_continuations_of_expr
          (if Pos.eqb y x || Pos.eqb y f
           then body
           else share_reads p x f same' body)
        = free_continuations_of_expr body).
    { destruct (Pos.eqb_spec y x) as [-> | Hyx]; cbn [orb];
        [ split; [ reflexivity
                 | split; [ apply incl_tl, incl_refl | reflexivity ] ] | ].
      destruct (Pos.eqb_spec y f) as [-> | Hyf];
        [ split; [ reflexivity
                 | split; [ apply incl_tl, incl_refl | reflexivity ] ] | ].
      destruct (share_reads_scoped p x f body same') as (Hc & Hv & Hk).
      split; [ exact Hc | split; [ | exact Hk ] ].
      intros z Hz. apply in_without in Hz. destruct Hz as [Hz Hy].
      destruct (Hv z Hz) as [<- | Hin]; [ left; reflexivity | right ].
      apply in_without. split; assumption. }
    destruct Hbody as (Hc & Hv & Hk). scope.
    split; [ exact Hc | split; [ | exact Hk ] ].
    intros z Hz. apply in_app_or in Hz. destruct Hz as [Hz | Hz].
    + destruct (shared_named_free p x f same d z Hz) as [<- | Hin];
        [ left; reflexivity | right; apply in_or_app; left; exact Hin ].
    + destruct (Hv z Hz) as [<- | Hin];
        [ left; reflexivity | right; apply in_or_app; right; exact Hin ].
  - destruct (share_reads_scoped p x f body same) as (Hc & Hv & Hk).
    assert (Hhandler :
      code_closed
        (if among x params || among f params
         then handler
         else share_reads p x f (without params same) handler)
      = code_closed handler
      ∧ incl
          (without params
             (free_variables_of_expr
                (if among x params || among f params
                 then handler
                 else share_reads p x f (without params same) handler)))
          (f :: without params (free_variables_of_expr handler))
      ∧ free_continuations_of_expr
          (if among x params || among f params
           then handler
           else share_reads p x f (without params same) handler)
        = free_continuations_of_expr handler).
    { destruct (among x params) eqn:Hxp; cbn [orb];
        [ split; [ reflexivity
                 | split; [ apply incl_tl, incl_refl | reflexivity ] ] | ].
      destruct (among f params) eqn:Hfp;
        [ split; [ reflexivity
                 | split; [ apply incl_tl, incl_refl | reflexivity ] ] | ].
      destruct (share_reads_scoped p x f handler (without params same))
        as (Hc' & Hv' & Hk').
      split; [ exact Hc' | split; [ | exact Hk' ] ].
      intros z Hz. apply in_without in Hz. destruct Hz as [Hz Hparams].
      destruct (Hv' z Hz) as [<- | Hin]; [ left; reflexivity | right ].
      apply in_without. split; assumption. }
    destruct Hhandler as (Hc' & Hv' & Hk'). scope.
    rewrite Hc, Hc', Hk, Hk'.
    split; [ reflexivity | split; [ | reflexivity ] ].
    intros z Hz. apply in_app_or in Hz. destruct Hz as [Hz | Hz].
    + destruct (Hv' z Hz) as [<- | Hin];
        [ left; reflexivity | right; apply in_or_app; left; exact Hin ].
    + destruct (Hv z Hz) as [<- | Hin];
        [ left; reflexivity | right; apply in_or_app; right; exact Hin ].
  - destruct scrutinee as [z | sym | c];
      try (split; [ reflexivity
                  | split; [ apply incl_tl, incl_refl | reflexivity ] ]).
    destruct (among z same);
      [ | split; [ reflexivity
                 | split; [ apply incl_tl, incl_refl | reflexivity ] ] ].
    scope. split; [ reflexivity | split; [ | reflexivity ] ].
    intros y Hy. cbn [free_variables_of_simple app] in Hy |- *.
    destruct Hy as [<- | Hy]; [ left; reflexivity | right; right; exact Hy ].
Qed.

(* What is left of a handler once a read is removed and the others are
   shared. *)
Lemma shared_without_scoped (share : bool) (params : list variable)
    (found : read) (x : variable) (handler : expr) :
  removes (read_variable found) (read_kind found) (read_primitive found) x
    handler (read_without found) ->
  code_closed handler = true ->
  code_closed (read_without (with_reads_shared share params found)) = true
  ∧ incl
      (free_variables_of_expr
         (read_without (with_reads_shared share params found)))
      (read_variable (with_reads_shared share params found)
       :: free_variables_of_expr handler)
  ∧ incl
      (free_continuations_of_expr
         (read_without (with_reads_shared share params found)))
      (free_continuations_of_expr handler).
Proof.
  intros Hremoves Hclosed.
  destruct (removes_scoped _ _ _ _ _ _ Hremoves Hclosed)
    as (Hclosed' & Hv & Hk).
  rewrite with_reads_shared_variable. unfold with_reads_shared.
  destruct share; [ | split; [ exact Hclosed' | split; assumption ] ].
  destruct (nth_error params (read_position found)) as [x' | ];
    cbn [read_without]; [ | split; [ exact Hclosed' | split; assumption ] ].
  destruct
    (share_reads_scoped (read_primitive found) x' (read_variable found)
       (read_without found) [])
    as (Hc & Hv' & Hk').
  rewrite Hc, Hk'. split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros z Hz.
  destruct (Hv' z Hz) as [<- | Hin]; [ left; reflexivity | exact (Hv z Hin) ].
Qed.

(* One read made a parameter. *)
Lemma unbox_read_scoped
    (extend : facts -> list simple -> option (list simple))
    (names : list variable) (known : facts) (k : continuation)
    (params params' : list variable) (metadata metadata' : handler_metadata)
    (handler handler' body body' : expr) (found : read) :
  (∀ (names : list variable) (known : facts) (args args' : list simple),
    facts_within names known ->
    extend known args = Some args' ->
    incl (free_variables_of_simples args')
      (free_variables_of_simples args ++ names)) ->
  facts_within names known ->
  unbox_read extend known k params metadata handler body found
  = Some (params', metadata', handler', body') ->
  (code_closed handler = true ->
   code_closed (read_without found) = true
   ∧ incl (free_variables_of_expr (read_without found))
       (read_variable found :: free_variables_of_expr handler)
   ∧ incl (free_continuations_of_expr (read_without found))
       (free_continuations_of_expr handler)) ->
  scoped_under names
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params' metadata' handler') body').
Proof.
  intros Hextend Hknown Hstep Hwithout. unfold unbox_read in Hstep.
  destruct
    (may_take params metadata (read_variable found)
     && negb (var_free_b (read_variable found) handler));
    [ | discriminate ].
  destruct (extend_applications_by extend known k body) as [extended | ]
    eqn:Hbody; [ | discriminate ].
  simpl in Hstep. injection Hstep as <- <- <- <-.
  pose proof (extend_applications_by_scoped extend Hextend body names known
    k extended Hknown Hbody) as Hin_body.
  intros Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hwithout Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hin_body Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy].
    + left. apply in_or_app. left.
      apply in_without in Hy. destruct Hy as [Hy Hparams].
      apply in_without. split.
      * destruct (Hv y Hy) as [Heq | Hin]; [ | exact Hin ].
        exfalso. apply Hparams, in_or_app. right. left. exact Heq.
      * intros Hin. apply Hparams, in_or_app. left. exact Hin.
    + apply Hv', in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
  - apply incl_app_app; [ exact Hk | apply incl_without, Hk' ].
Qed.

Lemma unbox_reads_scoped (fuel : nat) :
  ∀ (names : list variable) (known : facts) (k : continuation)
    (made : list (variable * nat)) (above : list (nat * list nat))
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr),
    facts_within names known ->
    scoped_under names
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (unbox_reads fuel known k made above params metadata handler body).
Proof.
  induction fuel as [ | fuel IH];
    intros names known k made above params metadata handler body Hknown;
    cbn [unbox_reads]; [ apply scoped_under_refl | ].
  destruct (next_read params handler) as [found0 | ] eqn:Hnext;
    cbn [option_map]; [ | apply scoped_under_refl ].
  set (share :=
    shares_reads known k (tests_of (read_position found0) made above) params
      handler body found0).
  set (found := with_reads_shared share params found0).
  destruct
    (loads_of_two_tags known k (tests_of (read_position found) made above)
       params handler body found);
    [ apply scoped_under_refl | ].
  set (made' :=
    if reads_test (read_primitive found)
    then (read_variable found, read_position found) :: made
    else made).
  set (above' :=
    (length params, length params :: boxes_of (read_position found) above)
    :: above).
  set (variant := tested_is_int params (read_position found) handler).
  set (tests := tests_of (read_position found) made above).
  assert (Hcertain :
    scoped_under names
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      match
        unbox_read
          (fun known =>
             extend_args_certain variant tests params handler
               (read_without found) known (read_position found)
               (read_primitive found) (read_kind found))
          known k params metadata handler body found
      with
      | Some (params0, metadata0, handler0, body0) =>
        unbox_reads fuel known k made' above' params0 metadata0 handler0
          body0
      | None =>
        Let_cont_non_recursive k (Handler params metadata handler) body
      end).
  { destruct
      (unbox_read
         (fun known =>
            extend_args_certain variant tests params handler
              (read_without found) known (read_position found)
              (read_primitive found) (read_kind found))
         known k params metadata handler body found)
      as [[[[params' metadata'] handler'] body'] | ] eqn:Hstep;
      [ | apply scoped_under_refl ].
    eapply scoped_under_trans; [ | apply IH; exact Hknown ].
    destruct (next_read_first params handler found0 Hnext)
      as [wanted Hfirst].
    destruct (first_read_removes handler wanted params [] found0 Hfirst)
      as (x & _ & _ & _ & Hremoves).
    exact (unbox_read_scoped _ names known k params params' metadata
      metadata' handler handler' body body' found
      (extend_args_certain_free variant tests params handler
         (read_without found) (read_position found) (read_primitive found)
         (read_kind found))
      Hknown Hstep
      (shared_without_scoped share params found0 x handler Hremoves)). }
  destruct (head_read params handler) as [head | ] eqn:Hhead;
    [ | exact Hcertain ].
  destruct (Pos.eqb (read_variable head) (read_variable found));
    [ | exact Hcertain ].
  destruct
    (unbox_read
       (fun known =>
          extend_args known (read_position head) (read_primitive head))
       known k params metadata handler body
       (with_reads_shared share params head))
    as [[[[params' metadata'] handler'] body'] | ] eqn:Hstep;
    [ | apply scoped_under_refl ].
  eapply scoped_under_trans; [ | apply IH; exact Hknown ].
  unfold head_read in Hhead.
  destruct handler as [ f kind d rest | | | | | | | ]; try discriminate.
  destruct (param_read reads_any params [] f d) as [[p i] | ] eqn:Hread;
    [ | discriminate ].
  simpl in Hhead. injection Hhead as <-.
  destruct (param_read_spec reads_any params [] f d p i Hread)
    as (x & -> & _).
  exact (unbox_read_scoped _ names known k params params' metadata
    metadata' _ handler' body body' _ (extend_args_free i p) Hknown Hstep
    (shared_without_scoped share params (Read f kind p i rest) x _
       (Removes_here f kind p x rest))).
Qed.

Fixpoint unbox_variants_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : facts),
    facts_within names known ->
    scoped_under names e (unbox_variants_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown; cbn [unbox_variants_knowing];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply incl_appl, incl_refl | ].
    apply unbox_variants_knowing_scoped, facts_under_let_within, Hknown.
  - apply scoped_under_let_static.
    apply unbox_variants_knowing_scoped, facts_under_statics_within, Hknown.
  - eapply scoped_under_trans; [ | apply unbox_reads_scoped, Hknown ].
    apply scoped_under_let_cont; apply unbox_variants_knowing_scoped;
      [ apply facts_under_params_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply unbox_variants_knowing_scoped, Hknown.
Qed.

Theorem unbox_variant_params_keeps_scope :
  unbox_variant_params_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e.
  apply scoped_under_none, unbox_variants_knowing_scoped.
  split; [ split; intros entry [] | split; intros entry [] ].
Qed.

(* ======================================================================== *)
(* The reads of a loop keep scope                                           *)
(* ======================================================================== *)

(* A read of a loop made a parameter: what is left of the handler
   mentions what the handler did and the new parameter, which it now
   binds; an application is given something in scope where it stands,
   in the handler the parameter read and the new one among them. *)
Lemma unbox_loop_read_scoped (names : list variable) (known : facts)
    (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler body : expr) (found : read) (x : variable)
    (state : loop_state) :
  facts_within names known ->
  nth_error params (read_position found) = Some x ->
  removes (read_variable found) (read_kind found) (read_primitive found) x
    handler (read_without found) ->
  unbox_loop_read known k params metadata handler body found = Some state ->
  scoped_under names (loop_of k (params, metadata, handler, body))
    (loop_of k state).
Proof.
  intros Hknown Hx Hremoves Hstep.
  destruct found as [f kind p i rest]. simpl in Hx, Hremoves.
  unfold unbox_loop_read in Hstep. simpl in Hstep. rewrite Hx in Hstep.
  simpl in Hstep.
  destruct
    (may_take params metadata f && negb (var_free_b f handler)
     && on_spine f handler);
    [ | discriminate ].
  destruct (extend_applications known k i p body) as [body' | ] eqn:Hbody;
    [ | discriminate ].
  simpl in Hstep.
  destruct
    (extend_applications
       (facts_under_let (facts_under_params known params) f
          (Prim (Unary p) [Var x]))
       k i p rest)
    as [rest' | ] eqn:Hrest; [ | discriminate ].
  simpl in Hstep. injection Hstep as <-.
  pose proof (extend_applications_scoped body names known k i p body' Hknown
    Hbody) as Hin_body.
  pose proof (extend_applications_scoped rest _ _ k i p rest'
    (facts_under_let_within names _ f (Prim (Unary p) [Var x])
       (facts_under_params_within names known params Hknown))
    Hrest) as Hin_handler.
  unfold loop_of. intros Hclosed. scope_in Hclosed. scope.
  cbn [forallb flat_map map fst] in Hclosed |- *.
  scope_in Hclosed. scope.
  rewrite ?app_nil_r, ?andb_true_r in Hclosed.
  rewrite ?app_nil_r, ?andb_true_r.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hbody_closed].
  destruct (removes_scoped f kind p x handler rest Hremoves Hfirst)
    as (Hfirst0 & Hv0 & Hk0).
  destruct (Hin_handler Hfirst0) as (Hfirst' & Hv & Hk).
  destruct (Hin_body Hbody_closed) as (Hbody_closed' & Hv' & Hk').
  rewrite Hfirst', Hbody_closed'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy].
    + apply in_without in Hy. destruct Hy as [Hy Hparams].
      apply Hv, in_app_or in Hy.
      destruct Hy as [Hy | [Heq | Hy]].
      * destruct (Hv0 y Hy) as [Heq | Hin];
          [ exfalso; apply Hparams, in_or_app; right; left; exact Heq | ].
        left. apply in_or_app. left. apply in_without.
        split; [ exact Hin | ].
        intros Hin'. apply Hparams, in_or_app. left. exact Hin'.
      * exfalso. apply Hparams, in_or_app. right. left. exact Heq.
      * apply in_app_or in Hy.
        destruct Hy as [Hy | Hy]; [ right; exact Hy | ].
        cbn in Hy. destruct Hy as [<- | []].
        exfalso. apply Hparams, in_or_app. left.
        exact (nth_error_In params i Hx).
    + apply Hv', in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
  - apply incl_without, incl_app_app;
      [ exact (incl_tran Hk Hk0) | exact Hk' ].
Qed.

Lemma unbox_loop_param_scoped (fuel : nat) :
  ∀ (names : list variable) (known : facts) (k : continuation) (i : nat)
    (state : loop_state),
    facts_within names known ->
    scoped_under names (loop_of k state)
      (loop_of k (unbox_loop_param fuel known k i state)).
Proof.
  induction fuel as [ | fuel IH]; intros names known k i state Hknown;
    [ apply scoped_under_refl | ].
  destruct state as [[[params metadata] handler] body].
  cbn [unbox_loop_param].
  destruct (nth_error params i) as [param | ] eqn:Hparam;
    [ | apply scoped_under_refl ].
  destruct (next_read [param] handler) as [r | ] eqn:Hnext;
    [ | apply scoped_under_refl ].
  destruct
    (unbox_loop_read known k params metadata handler body
       (Read (read_variable r) (read_kind r) (read_primitive r) i
          (read_without r)))
    as [state' | ] eqn:Hread; [ | apply scoped_under_refl ].
  eapply scoped_under_trans; [ | apply IH; exact Hknown ].
  destruct (next_read_first [param] handler r Hnext) as [wanted Hfirst].
  destruct (first_read_removes handler wanted [param] [] r Hfirst)
    as (x & Hposition & _ & _ & Hremoves).
  apply position_single in Hposition. subst x.
  exact (unbox_loop_read_scoped names known k params metadata handler body
    (Read (read_variable r) (read_kind r) (read_primitive r) i
       (read_without r))
    param state' Hknown Hparam Hremoves Hread).
Qed.

Lemma unbox_loop_param_all_scoped (fuel : nat) (names : list variable)
    (known : facts) (k : continuation) (i : nat) (state : loop_state) :
  facts_within names known ->
  scoped_under names (loop_of k state)
    (loop_of k (unbox_loop_param_all fuel known k i state)).
Proof.
  intros Hknown. unfold unbox_loop_param_all.
  destruct (reads_left i (unbox_loop_param fuel known k i state));
    [ apply scoped_under_refl | apply unbox_loop_param_scoped, Hknown ].
Qed.

Lemma unbox_loop_params_scoped (fuel : nat) (names : list variable)
    (known : facts) (k : continuation) (positions : list nat) :
  ∀ state : loop_state,
    facts_within names known ->
    scoped_under names (loop_of k state)
      (loop_of k (unbox_loop_params fuel known k positions state)).
Proof.
  induction positions as [ | i positions IH]; intros state Hknown;
    cbn [unbox_loop_params]; [ apply scoped_under_refl | ].
  eapply scoped_under_trans;
    [ apply unbox_loop_param_all_scoped, Hknown | apply IH, Hknown ].
Qed.

Fixpoint unbox_loops_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : facts),
    facts_within names known ->
    scoped_under names e (unbox_loops_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown; cbn [unbox_loops_knowing];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply incl_appl, incl_refl | ].
    apply unbox_loops_knowing_scoped, facts_under_let_within, Hknown.
  - apply scoped_under_let_static.
    apply unbox_loops_knowing_scoped, facts_under_statics_within, Hknown.
  - apply scoped_under_let_cont; apply unbox_loops_knowing_scoped;
      [ apply facts_under_params_within, Hknown | exact Hknown ].
  - assert (Hbody :
      scoped_under names (Let_cont_recursive handlers body)
        (Let_cont_recursive handlers (unbox_loops_knowing known body)))
      by (apply scoped_under_let_cont_recursive, unbox_loops_knowing_scoped,
            Hknown).
    destruct handlers as [ | [k [params metadata handler]] [ | h2 handlers]];
      try exact Hbody.
    eapply scoped_under_trans; [ exact Hbody | ].
    exact (unbox_loop_params_scoped _ names known k _
      (params, metadata, handler, unbox_loops_knowing known body) Hknown).
Qed.
