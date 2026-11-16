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

(* Proof of the statement of FlambdaDemoteScopeSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaWellFormedSpec FlambdaValidatorEquivarianceSpec
  FlambdaScopedProofs FlambdaDemoteSpec FlambdaDemoteScopeSpec.
Import ListNotations.

(* An [Apply_cont] without its trap action has the same arguments, and
   its continuations are among those it had. *)
Lemma strip_trap_args (k : continuation) (ac : apply_cont) :
  ac_args (strip_trap k ac) = ac_args ac.
Proof. unfold strip_trap. destruct (traps k ac); reflexivity. Qed.

Lemma strip_trap_continuations (k : continuation) (ac : apply_cont) :
  incl (apply_cont_continuations (strip_trap k ac))
    (apply_cont_continuations ac).
Proof.
  unfold strip_trap. destruct (traps k ac); [ | apply incl_refl ].
  unfold apply_cont_continuations. simpl. intros k' [<- | [ ]].
  left. reflexivity.
Qed.

Lemma scoped_apply_cont_strip (k : continuation) (ac : apply_cont) :
  scoped (Apply_cont ac) (Apply_cont (strip_trap k ac)).
Proof.
  intros _. scope. unfold free_variables_of_apply_cont.
  rewrite strip_trap_args.
  split; [ reflexivity | ].
  split; [ apply incl_refl | apply strip_trap_continuations ].
Qed.

Lemma scoped_switch_strip (k : continuation) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  scoped (Switch scrutinee arms)
    (Switch scrutinee
       (map (fun arm => (fst arm, strip_trap k (snd arm))) arms)).
Proof.
  intros _. scope. split; [ reflexivity | ]. split.
  - apply incl_app; [ apply incl_appl, incl_refl | apply incl_appr ].
    induction arms as [ | [i ac] arms IH]; cbn [map flat_map fst snd];
      [ apply incl_refl | ].
    unfold free_variables_of_apply_cont at 1 3. rewrite strip_trap_args.
    apply incl_app; [ apply incl_appl, incl_refl | apply incl_appr, IH ].
  - induction arms as [ | [i ac] arms IH]; cbn [map flat_map fst snd];
      [ apply incl_refl | ].
    apply incl_app;
      [ apply incl_appl, strip_trap_continuations | apply incl_appr, IH ].
Qed.

Lemma map_fst_same (f : continuation_handler -> continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  map fst (map (fun kh => (fst kh, f (snd kh))) handlers) = map fst handlers.
Proof. rewrite map_map. apply map_ext. intros [k h]. reflexivity. Qed.

Fixpoint scoped_strip_traps (k : continuation) (e : expr) :
  scoped e (strip_traps k e)
with scoped_strip_traps_handler (k : continuation)
    (h : continuation_handler) :
  scoped_handler h (strip_traps_handler k h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [strip_traps].
    + apply scoped_let; [ apply incl_refl | apply scoped_strip_traps ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply scoped_strip_traps ].
    + apply scoped_let_cont;
        [ apply scoped_strip_traps_handler | apply scoped_strip_traps ].
    + apply scoped_let_cont_recursive;
        [ apply map_fst_same | | apply scoped_strip_traps ].
      induction handlers as [ | [k' h] handlers IH]; cbn [map fst snd];
        [ apply scoped_handlers_nil | ].
      apply scoped_handlers_cons;
        [ apply scoped_strip_traps_handler | exact IH ].
    + apply scoped_refl.
    + apply scoped_apply_cont_strip.
    + apply scoped_switch_strip.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. cbn [strip_traps_handler].
    apply scoped_handler_intro, scoped_strip_traps.
Qed.

Lemma scoped_handler_cleared (h h' : continuation_handler) :
  scoped_handler h h' -> scoped_handler h (clear_exn_handler h').
Proof.
  destruct h as [params metadata handler],
    h' as [params' metadata' handler'].
  intros Hh Hclosed. exact (Hh Hclosed).
Qed.

Fixpoint scoped_demote_handlers (e : expr) :
  ∀ scope : list continuation, scoped e (demote_handlers scope e)
with scoped_demote_handlers_handler (h : continuation_handler) :
  ∀ scope : list continuation,
    scoped_handler h (demote_handlers_handler scope h)
with scoped_demote_handlers_code (c : code) :
  scoped_code c (demote_handlers_code c).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros scope; cbn [demote_handlers]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply scoped_demote_handlers ].
    + apply scoped_let_static; [ | apply scoped_demote_handlers ].
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ apply scoped_bindings_nil | ].
      destruct binding as [id c | id | s const];
        cbn [demote_handlers_binding];
        [ apply scoped_bindings_cons_code;
          [ apply scoped_demote_handlers_code | exact IH ]
        | apply scoped_bindings_cons_same; exact IH
        | apply scoped_bindings_cons_same; exact IH ].
    + destruct (demotable scope k h body).
      * apply scoped_let_cont.
        -- apply scoped_handler_cleared, scoped_demote_handlers_handler.
        -- eapply scoped_trans;
             [ apply scoped_demote_handlers | apply scoped_strip_traps ].
      * apply scoped_let_cont;
          [ apply scoped_demote_handlers_handler
          | apply scoped_demote_handlers ].
    + generalize (map fst handlers ++ scope). intros inner.
      apply scoped_let_cont_recursive;
        [ apply map_fst_same | | apply scoped_demote_handlers ].
      induction handlers as [ | [k h] handlers IH]; cbn [map fst snd];
        [ apply scoped_handlers_nil | ].
      apply scoped_handlers_cons;
        [ apply scoped_demote_handlers_handler | exact IH ].
  - destruct h as [params metadata handler]. intros scope.
    cbn [demote_handlers_handler].
    apply scoped_handler_intro, scoped_demote_handlers.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [demote_handlers_code].
    apply scoped_code_intro, scoped_demote_handlers.
Qed.

Theorem demote_handlers_keeps_scope : demote_handlers_keeps_scope_statement.
Proof.
  intros scope. apply keeps_scope_intro. intros e.
  apply scoped_demote_handlers.
Qed.
