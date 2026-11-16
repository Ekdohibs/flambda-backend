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

(* Proofs of the statements of FlambdaForwardJumpsSpec.v. Nothing here
   needs review.

   The pass is FlambdaForwardSpec.forward_applications with fewer
   entries, and the relation between a term and its result is that of
   FlambdaForwardProofs.v, taken strict: every entry is for a
   continuation of no parameters whose handler gives no kinds
   ([jumps]), so an application that goes on has no arguments, which
   are trivially of the kinds of the parameters. An application of the
   first term that goes on is then defined where the [Apply_cont] it
   becomes is: there is no argument to evaluate, no kind to test and
   nothing to bind. So the first run is undefined only where the
   second is, the rounds of FlambdaForwardProofs.v are read both ways
   ([fwd_round], [fwd_round_back]), and each term refines the other. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaRoundsProofs FlambdaUnusedParamsSpec
  FlambdaUnusedParamsProofs FlambdaParamAliasesSpec FlambdaParamAliasesProofs
  FlambdaForwardSpec FlambdaForwardProofs
  FlambdaForwardJumpsSpec.
From OxCamlFlambda2 Require Import FlambdaMapProofs.
Import ListNotations.

Fixpoint forward_jumps_binds_statics (e : expr) :
  ∀ F : forwards,
    binds_statics (forward_jumps_under F e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros F; simpl; try reflexivity.
  - apply forward_jumps_binds_statics.
  - rewrite forward_jumps_binds_statics. reflexivity.
  - rewrite !forward_jumps_binds_statics. reflexivity.
  - rewrite forward_jumps_binds_statics. f_equal.
    generalize (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
                  handlers
                ++ F).
    intros F'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite forward_jumps_binds_statics, IH. reflexivity.
Qed.

(* The entry that the pass makes is a strict one for the handler, which
   the body leaves alone. *)
Lemma jump_entry_fits (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler' body : expr) :
  entry_fits true k params metadata handler'
    (jump_entry k params metadata handler' body)
  ∧ (∀ x : variable,
      frozen_var (jump_entry k params metadata handler' body) x ->
      binds_var x body = false)
  ∧ (∀ k0 : continuation,
      frozen_cont (jump_entry k params metadata handler' body) k0 ->
      binds_cont k0 body = false).
Proof.
  unfold jump_entry.
  assert (Hnone :
    entry_fits true k params metadata handler' None
    ∧ (∀ x : variable, frozen_var None x -> binds_var x body = false)
    ∧ (∀ k0 : continuation, frozen_cont None k0 -> binds_cont k0 body = false)).
  { split; [ exact I | ].
    split; [ intros x (params0 & target & Hsome & _)
           | intros k0 (params0 & target & Hsome & _) ];
      discriminate Hsome. }
  destruct params as [ | p params]; [ | exact Hnone ].
  destruct (hm_params_kinds metadata) eqn:Hkinds; [ | exact Hnone ].
  apply forwarding_fits. intros _. rewrite Hkinds. reflexivity.
Qed.

(* Every entry is for a continuation of no parameters whose handler
   gives no kinds. *)
Definition jumps (K : kinds_of_conts) (F : forwards) : Prop :=
  ∀ (k : continuation) (params : list variable) (target : apply_cont),
    forward_of F k = Some (params, target) ->
    params = [] ∧ lookup k K = Some [].

Lemma jump_entry_some (k : continuation) (params params' : list variable)
    (metadata : handler_metadata) (handler body : expr)
    (target : apply_cont) :
  jump_entry k params metadata handler body = Some (params', target) ->
  params' = [] ∧ hm_params_kinds metadata = [].
Proof.
  unfold jump_entry, forwarding.
  destruct params as [ | p params]; [ | discriminate ].
  destruct (hm_params_kinds metadata); [ | discriminate ].
  destruct handler; try discriminate.
  match goal with
  | |- (if ?check then _ else _) = _ -> _ =>
    destruct check; [ discriminate | ]
  end.
  intros Hsome. injection Hsome as <- _. split; reflexivity.
Qed.

Lemma jumps_cons (K : kinds_of_conts) (F : forwards) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  jumps K F ->
  jumps ((k, hm_params_kinds metadata) :: K)
    ((k, jump_entry k params metadata handler body) :: F).
Proof.
  intros Hjumps k' params' target Hsome.
  destruct (Pos.eq_dec k' k) as [-> | Hne].
  - rewrite forward_of_head in Hsome.
    destruct (jump_entry_some _ _ _ _ _ _ _ Hsome) as [-> Hkinds].
    split; [ reflexivity | ]. rewrite lookup_head, Hkinds. reflexivity.
  - rewrite forward_of_other in Hsome by exact Hne.
    rewrite lookup_other by exact Hne. exact (Hjumps k' params' target Hsome).
Qed.

Lemma jumps_plain (K : kinds_of_conts) (F : forwards)
    (handlers : list (continuation * continuation_handler)) :
  jumps K F -> jumps K (plain_forwards handlers ++ F).
Proof.
  intros Hjumps k params target Hsome. rewrite forward_of_plain in Hsome.
  destruct (lookup k handlers); [ discriminate Hsome | ].
  exact (Hjumps k params target Hsome).
Qed.

(* An application that goes on has no arguments. *)
Lemma goes_jump (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
    (ac : apply_cont) :
  jumps K F -> goes true G K F ac (forward_apply_cont F ac).
Proof.
  intros Hjumps. unfold forward_apply_cont.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ] eqn:Hk;
    [ | apply Goes_same ].
  destruct
    (Nat.eqb (length (ac_args ac)) (length params) && no_trap_action ac
     && no_trap_action target)
    eqn:Hcheck; [ | apply Goes_same ].
  apply andb_true_iff in Hcheck as [Hcheck _].
  apply andb_true_iff in Hcheck as [Hlength _].
  apply Nat.eqb_eq in Hlength.
  destruct (Hjumps _ _ _ Hk) as [-> Hkinds].
  apply Goes_on; [ exact Hk | exact Hlength | ].
  intros _. exists []. split; [ exact Hkinds | ].
  destruct (ac_args ac); [ | discriminate Hlength ].
  intros vars _. exists []. split; reflexivity.
Qed.

Fixpoint forward_jumps_fwd (e : expr) :
  ∀ (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards),
    safe F e -> jumps K F -> fwd true G K F e (forward_jumps_under F e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros G K F Hsafe Hjumps; simpl.
  - destruct (safe_let F x kind d body Hsafe) as [Hbody Hx].
    apply Fwd_let; [ exact Hx | apply forward_jumps_fwd; assumption ].
  - apply Fwd_let_static, forward_jumps_fwd; assumption.
  - destruct (safe_let_cont F k params metadata handler body Hsafe)
      as (Hhandler & Hbody & Hparams & Hk).
    destruct
      (jump_entry_fits k params metadata
         (forward_jumps_under F handler) body)
      as (Hfits & Hvars & Hconts).
    apply (Fwd_let_cont true G K F k params metadata metadata handler
      (forward_jumps_under F handler)
      (jump_entry k params metadata (forward_jumps_under F handler)
         body));
      [ apply forward_jumps_fwd; assumption | reflexivity
      | exact Hparams | exact Hk | exact Hfits | ].
    apply forward_jumps_fwd; [ apply safe_cons; assumption | ].
    apply jumps_cons, Hjumps.
  - destruct (safe_let_cont_recursive F handlers body Hsafe)
      as [Hbody Hhandlers].
    change (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
              handlers)
      with (plain_forwards handlers).
    assert (Hall :
      Forall
        (fun kh =>
           safe (plain_forwards handlers ++ F) (handler_expr (snd kh))
           ∧ (∀ p : variable, p ∈ handler_params (snd kh) -> var_unfrozen F p)
           ∧ cont_unfrozen F (fst kh))
        handlers).
    { apply Forall_forall. intros [k [params metadata handler]] Hin.
      destruct (Hhandlers k params metadata handler Hin)
        as (Hhandler & Hparams & Hk).
      split; [ apply safe_plain, Hhandler | split; assumption ]. }
    pose proof (jumps_plain K F handlers Hjumps) as Hjumps'.
    apply Fwd_let_cont_recursive.
    + clear Hsafe Hhandlers. revert Hall Hjumps'.
      generalize (plain_forwards handlers ++ F). intros F' Hall Hjumps'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        [ apply Forall2_nil | ].
      inversion Hall as [ | kh rest (Hhandler & Hparams & Hk) Hrest].
      subst kh rest.
      apply Forall2_cons; [ | exact (IH Hrest) ].
      simpl. split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ].
      split; [ apply forward_jumps_fwd; assumption | ].
      split; assumption.
    + clear. unfold handlers_bind_statics.
      generalize (plain_forwards handlers ++ F). intros F'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl; [ reflexivity | ].
      rewrite forward_jumps_binds_statics, IH. reflexivity.
    + apply forward_jumps_fwd; [ apply safe_plain, Hbody | exact Hjumps' ].
  - apply Fwd_apply, Returns_same.
  - apply Fwd_apply_cont, goes_jump, Hjumps.
  - apply Fwd_switch. clear Hsafe.
    induction arms as [ | arm arms IH]; simpl; [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact IH ].
    split; [ reflexivity | apply goes_jump, Hjumps ].
  - apply Fwd_invalid.
Qed.

Theorem forward_jumps_equivalent : forward_jumps_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  assert (Hrel :
    state_fwd true (State e vars conts statics hp rd)
      (State (forward_jumps e) vars conts statics hp rd)).
  { apply (State_fwd true [] [] []);
      [ apply forward_jumps_fwd;
          [ apply safe_none; reflexivity
          | intros k params target Hsome; discriminate Hsome ]
      | intros k; apply Same_fwd | apply held_nil | apply kinds_hold_nil ]. }
  apply refines_antisymmetric.
  - exact (rounds_refines (state_fwd true) (fwd_round true) _ _ Hrel).
  - exact
      (rounds_refines (fun t' t => state_fwd true t t') fwd_round_back _ _
         Hrel).
Qed.
