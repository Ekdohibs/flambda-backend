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


(* Proofs of the statements of FlambdaUnboxInScopeRenameSpec.v. Nothing
   here needs review.

   The validation runs the two rewrites:
   FlambdaValidatorEquivarianceProofs.v and FlambdaValidateScopeProofs.v
   need what is proved here for the round of FlambdaValidateSpec.v, so
   this file imports neither. The two proofs of each rewrite are those
   of FlambdaNormaliseSpec.rewrite_code and of
   FlambdaRecHandlersSpec.rewrite_rec_handlers, with what is known of
   symbols alone given to the rewrite. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaNormaliseSpec FlambdaBlockLoadSpec FlambdaUnboxSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaKeepsScopeProofs FlambdaUnboxParamsSpec FlambdaUnboxParamsRenameSpec
  FlambdaUnboxVariantsSpec FlambdaUnboxInScopeSpec FlambdaUnboxInScopeProofs
  FlambdaUnboxInScopeRenameSpec FlambdaUnboxLoopsSpec
  FlambdaUnboxTogetherSpec.
From OxCamlFlambda2 Require FlambdaUnboxParamsNamesProofs.
From OxCamlFlambda2 Require FlambdaUnboxTogetherNamesProofs.
Import ListNotations.

(* ======================================================================== *)
(* What is known of symbols, and renaming                                   *)
(* ======================================================================== *)

Theorem symbols_only_renamed : symbols_only_renamed_statement.
Proof.
  intros r known Honly. rewrite <- Honly.
  unfold rename_facts, facts_of_symbols, rename_known_blocks,
    rename_known_numbers, map_values.
  cbn [facts_blocks facts_numbers known_vars known_symbols
       known_wrapped_vars known_boxed_symbols map].
  do 2 f_equal. rewrite map_map. apply map_ext. intros [s [tag fields]].
  unfold rename_known_block. cbn [fst snd]. rewrite map_map.
  apply f_equal, f_equal, map_ext.
  intros [[x | s' | c] | ]; reflexivity.
Qed.

Lemma closed_blocks_renamed (r : renaming) (bindings : list static_binding) :
  map
    (fun entry : symbol * known_block =>
       (fst entry, (fst (snd entry), map closed_field (snd (snd entry)))))
    (flat_map static_block_made (map (rename_static_binding r) bindings))
  = map
      (fun entry : symbol * known_block =>
         (fst entry, (fst (snd entry), map closed_field (snd (snd entry)))))
      (flat_map static_block_made bindings).
Proof.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite !map_app, IH. f_equal.
  destruct binding
    as [id c | id
       | s [code slots | fslot syms decls slots | tag fields | contents
           | bytes | elements]];
    try reflexivity.
  cbn. do 3 f_equal. rewrite !map_map. apply map_ext.
  intros [x | s' | c]; reflexivity.
Qed.

Lemma boxed_numbers_renamed (r : renaming) (bindings : list static_binding) :
  flat_map static_number_boxed (map (rename_static_binding r) bindings)
  = flat_map static_number_boxed bindings.
Proof.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite IH. f_equal.
  destruct binding
    as [id c | id
       | s [code slots | fslot syms decls slots | tag fields | contents
           | bytes | elements]];
    try reflexivity.
  destruct contents as [x | s' | c]; reflexivity.
Qed.

Theorem symbols_under_statics_renamed :
  symbols_under_statics_renamed_statement.
Proof.
  intros r known bindings.
  unfold symbols_under_statics, facts_under_statics, known_under_statics,
    numbers_under_statics, facts_of_symbols.
  cbn [facts_blocks facts_numbers known_symbols known_boxed_symbols].
  rewrite !map_app, closed_blocks_renamed, boxed_numbers_renamed.
  reflexivity.
Qed.

(* ======================================================================== *)
(* The two rewrites commute with renaming                                   *)
(* ======================================================================== *)

Section Renamed.

Variable f : facts -> expr -> expr.

Hypothesis f_equivariant : equivariant_where_known f.

Lemma rewritten_body_renamed (known : facts) (r : renaming) (body : expr) :
  symbols_only known ->
  rewrite_code_body (f known) (rename_expr r body)
  = rename_expr r (rewrite_code_body (f known) body).
Proof.
  intros Honly. unfold rewrite_code_body.
  rewrite (f_equivariant known Honly), !rename_binds_statics.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f known body)));
    reflexivity.
Qed.

Lemma rewritten_binding_renamed (known : facts) (r : renaming)
    (binding : static_binding) :
  symbols_only known ->
  rewrite_code_in_binding (f known) (rename_static_binding r binding)
  = rename_static_binding r (rewrite_code_in_binding (f known) binding).
Proof.
  intros Honly.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    try reflexivity.
  cbn [rename_static_binding rename_code rewrite_code_in_binding].
  rewrite (rewritten_body_renamed known _ _ Honly). reflexivity.
Qed.

Fixpoint rewrite_code_in_scope_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : facts,
    rewrite_code_in_scope f known (rename_expr r e)
    = rename_expr r (rewrite_code_in_scope f known e)
with rewrite_code_in_scope_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : facts,
    rewrite_code_in_scope_in_handler f known (rename_handler r h)
    = rename_handler r (rewrite_code_in_scope_in_handler f known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; cbn [rename_expr rewrite_code_in_scope]; cbv zeta;
      try reflexivity.
    + rewrite rewrite_code_in_scope_equivariant_aux. reflexivity.
    + rewrite symbols_under_statics_renamed,
        rewrite_code_in_scope_equivariant_aux, !map_map.
      f_equal. apply map_ext. intros binding.
      apply rewritten_binding_renamed, facts_of_symbols_only.
    + rewrite rewrite_code_in_scope_in_handler_equivariant,
        rewrite_code_in_scope_equivariant_aux.
      reflexivity.
    + rewrite rewrite_code_in_scope_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (rewrite_code_in_scope_in_handler_equivariant r h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known.
    cbn [rename_handler rewrite_code_in_scope_in_handler].
    rewrite rewrite_code_in_scope_equivariant_aux. reflexivity.
Qed.

Fixpoint rewrite_rec_handlers_in_scope_equivariant_aux (r : renaming)
    (e : expr) :
  ∀ known : facts,
    symbols_only known ->
    rewrite_rec_handlers_in_scope f known (rename_expr r e)
    = rename_expr r (rewrite_rec_handlers_in_scope f known e)
with rewrite_rec_handlers_in_scope_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : facts,
    symbols_only known ->
    rewrite_rec_handlers_in_scope_in_handler f known (rename_handler r h)
    = rename_handler r (rewrite_rec_handlers_in_scope_in_handler f known h)
with rewrite_rec_handler_in_scope_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : facts,
    symbols_only known ->
    rewrite_rec_handler_in_scope f known (rename_handler r h)
    = rename_handler r (rewrite_rec_handler_in_scope f known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known Honly; cbn [rename_expr rewrite_rec_handlers_in_scope];
      try reflexivity.
    + rewrite rewrite_rec_handlers_in_scope_equivariant_aux by exact Honly.
      reflexivity.
    + rewrite symbols_under_statics_renamed,
        rewrite_rec_handlers_in_scope_equivariant_aux
        by apply facts_of_symbols_only.
      reflexivity.
    + rewrite rewrite_rec_handlers_in_scope_in_handler_equivariant,
        rewrite_rec_handlers_in_scope_equivariant_aux
        by exact Honly.
      reflexivity.
    + rewrite rewrite_rec_handlers_in_scope_equivariant_aux by exact Honly.
      f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (rewrite_rec_handler_in_scope_equivariant r h known Honly), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known Honly.
    cbn [rename_handler rewrite_rec_handlers_in_scope_in_handler].
    rewrite rewrite_rec_handlers_in_scope_equivariant_aux by exact Honly.
    reflexivity.
  - destruct h as [params metadata handler]. intros known Honly.
    cbn [rename_handler rewrite_rec_handler_in_scope].
    rewrite rewrite_rec_handlers_in_scope_equivariant_aux by exact Honly.
    rewrite (rewritten_body_renamed known _ _ Honly). reflexivity.
Qed.

End Renamed.

Theorem rewrite_code_in_scope_equivariant :
  rewrite_code_in_scope_equivariant_statement.
Proof.
  intros f Hf known _ r e.
  exact (rewrite_code_in_scope_equivariant_aux f Hf r e known).
Qed.

Theorem rewrite_rec_handlers_in_scope_equivariant :
  rewrite_rec_handlers_in_scope_equivariant_statement.
Proof.
  intros f Hf known Honly r e.
  exact (rewrite_rec_handlers_in_scope_equivariant_aux f Hf r e known Honly).
Qed.

(* ======================================================================== *)
(* The two rewrites keep scope                                              *)
(* ======================================================================== *)

Section Scoped.

Variable f : facts -> expr -> expr.

Hypothesis f_keeps_scope : keeps_scope_where_known f.

Fixpoint rewrite_code_in_scope_scoped (e : expr) :
  ∀ known : facts, scoped e (rewrite_code_in_scope f known e)
with rewrite_code_in_scope_in_handler_scoped (h : continuation_handler) :
  ∀ known : facts,
    scoped_handler h (rewrite_code_in_scope_in_handler f known h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known;
      cbn [rewrite_code_in_scope rewrite_code_in_scope_in_handler];
      cbv zeta.
    + apply scoped_let;
        [ apply incl_refl | apply rewrite_code_in_scope_scoped ].
    + apply scoped_let_static;
        [ apply rewrite_code_in_bindings_scoped, f_keeps_scope,
            facts_of_symbols_only
        | apply rewrite_code_in_scope_scoped ].
    + apply scoped_let_cont;
        [ apply rewrite_code_in_scope_in_handler_scoped
        | apply rewrite_code_in_scope_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply rewrite_code_in_scope_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply rewrite_code_in_scope_in_handler_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros known.
    cbn [rewrite_code_in_scope rewrite_code_in_scope_in_handler].
    apply scoped_handler_intro, rewrite_code_in_scope_scoped.
Qed.

Fixpoint rewrite_rec_handlers_in_scope_scoped (e : expr) :
  ∀ known : facts,
    symbols_only known ->
    scoped e (rewrite_rec_handlers_in_scope f known e)
with rewrite_rec_handlers_in_scope_in_handler_scoped
    (h : continuation_handler) :
  ∀ known : facts,
    symbols_only known ->
    scoped_handler h (rewrite_rec_handlers_in_scope_in_handler f known h)
with rewrite_rec_handler_in_scope_scoped (h : continuation_handler) :
  ∀ known : facts,
    symbols_only known ->
    scoped_handler h (rewrite_rec_handler_in_scope f known h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known Honly;
      cbn [rewrite_rec_handlers_in_scope
           rewrite_rec_handlers_in_scope_in_handler
           rewrite_rec_handler_in_scope].
    + apply scoped_let;
        [ apply incl_refl
        | apply rewrite_rec_handlers_in_scope_scoped, Honly ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl
        | apply rewrite_rec_handlers_in_scope_scoped, facts_of_symbols_only ].
    + apply scoped_let_cont;
        [ apply rewrite_rec_handlers_in_scope_in_handler_scoped, Honly
        | apply rewrite_rec_handlers_in_scope_scoped, Honly ].
    + apply scoped_let_cont_recursive;
        [ | | apply rewrite_rec_handlers_in_scope_scoped, Honly ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply rewrite_rec_handler_in_scope_scoped, Honly | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros known Honly.
    cbn [rewrite_rec_handlers_in_scope
         rewrite_rec_handlers_in_scope_in_handler
         rewrite_rec_handler_in_scope].
    apply scoped_handler_intro, rewrite_rec_handlers_in_scope_scoped, Honly.
  - destruct h as [params metadata handler]. intros known Honly.
    cbn [rewrite_rec_handlers_in_scope
         rewrite_rec_handlers_in_scope_in_handler
         rewrite_rec_handler_in_scope].
    apply scoped_handler_intro.
    eapply scoped_trans;
      [ apply rewrite_rec_handlers_in_scope_scoped, Honly
      | apply rewrite_code_body_scoped, f_keeps_scope, Honly ].
Qed.

End Scoped.

Theorem rewrite_code_in_scope_keeps_scope :
  rewrite_code_in_scope_keeps_scope_statement.
Proof.
  intros f Hf known _. apply keeps_scope_intro. intros e.
  exact (rewrite_code_in_scope_scoped f Hf e known).
Qed.

Theorem rewrite_rec_handlers_in_scope_keeps_scope :
  rewrite_rec_handlers_in_scope_keeps_scope_statement.
Proof.
  intros f Hf known Honly. apply keeps_scope_intro. intros e.
  exact (rewrite_rec_handlers_in_scope_scoped f Hf e known Honly).
Qed.

(* ======================================================================== *)
(* The unboxing given what is known of symbols                              *)
(* ======================================================================== *)

(* What is known of symbols alone holds no variable. *)
Lemma symbols_only_within (known : facts) :
  symbols_only known ->
  FlambdaUnboxParamsNamesProofs.facts_within [] known.
Proof.
  intros Honly. rewrite <- Honly. split; [ split | split ].
  - intros entry [].
  - intros entry Hentry s Hs.
    cbn [facts_of_symbols facts_blocks known_symbols] in Hentry.
    apply in_map_iff in Hentry as ([s' [tag fields]] & <- & _).
    cbn [fst snd] in Hs. apply in_map_iff in Hs as (field & Hfield & _).
    destruct field as [[x | s'' | c] | ]; try discriminate Hfield;
      injection Hfield as <-; apply incl_refl.
  - intros entry [].
  - intros entry [].
Qed.

Theorem unbox_variants_knowing_symbols_keeps_scope :
  unbox_variants_knowing_symbols_keeps_scope_statement.
Proof.
  intros known Honly. apply keeps_scope_intro. intros e.
  apply scoped_under_none.
  apply FlambdaUnboxParamsNamesProofs.unbox_variants_knowing_scoped.
  exact (symbols_only_within known Honly).
Qed.

Theorem unbox_loops_knowing_symbols_keeps_scope :
  unbox_loops_knowing_symbols_keeps_scope_statement.
Proof.
  intros known Honly. apply keeps_scope_intro. intros e.
  apply scoped_under_none.
  apply FlambdaUnboxParamsNamesProofs.unbox_loops_knowing_scoped.
  exact (symbols_only_within known Honly).
Qed.

Theorem unbox_together_knowing_symbols_keeps_scope :
  unbox_together_knowing_symbols_keeps_scope_statement.
Proof.
  intros known Honly. apply keeps_scope_intro. intros e.
  apply scoped_under_none.
  apply FlambdaUnboxTogetherNamesProofs.unbox_together_knowing_scoped.
  exact (symbols_only_within known Honly).
Qed.
