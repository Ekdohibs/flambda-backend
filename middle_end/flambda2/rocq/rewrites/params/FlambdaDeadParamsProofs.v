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

(* Proofs of the statements of FlambdaDeadParamsSpec.v. Nothing here
   needs review.

   The pass is one more way of choosing the masks of the relation of
   FlambdaUnusedParamsProofs.v ([drops]), which is what that file's
   simulation asks: the second expression is the first with parameters
   and arguments removed as some masks say, the parameters that a mask
   removes not free in the second handler, and with [Let]s removed
   whose variables the second expression does not use. Here the masks
   of a loop are those that the search has found, or the masks that
   keep everything; either way they are the masks that one more round
   computes from them ([settle_fixed]), so each is [used_params] of its
   handler as it is rewritten under them, and that is a mask that fits
   it (FlambdaUnusedParamsProofs.used_params_fit). Nothing is proved of
   the search itself. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaRoundsProofs
  FlambdaUnusedParamsSpec FlambdaUnusedParamsProofs FlambdaDeadParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The search                                                               *)
(* ======================================================================== *)

Lemma bools_eqb_eq (l l' : list bool) : bools_eqb l l' = true -> l = l'.
Proof.
  revert l'.
  induction l as [ | b l IH]; intros [ | b' l']; simpl; try discriminate;
    [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hb Hl].
  apply Bool.eqb_prop in Hb. rewrite Hb, (IH l' Hl). reflexivity.
Qed.

Lemma mask_eqb_eq (mask mask' : option (list bool)) :
  mask_eqb mask mask' = true -> mask = mask'.
Proof.
  destruct mask as [mask | ], mask' as [mask' | ]; simpl; try discriminate;
    [ | reflexivity ].
  intros Heq. rewrite (bools_eqb_eq mask mask' Heq). reflexivity.
Qed.

Lemma masks_eqb_eq (known known' : masks) :
  masks_eqb known known' = true -> known = known'.
Proof.
  revert known'.
  induction known as [ | [k mask] known IH]; intros [ | [k' mask'] known'];
    simpl; try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Heq Hknown].
  apply andb_true_iff in Heq as [Hk Hmask]. apply Pos.eqb_eq in Hk.
  rewrite Hk, (mask_eqb_eq mask mask' Hmask), (IH known' Hknown).
  reflexivity.
Qed.

(* What the search finds is left as it is by one more round. *)
Lemma settle_fixed (fuel : nat) (next : masks -> masks) :
  ∀ loop found : masks,
    settle fuel next loop = Some found -> next found = found.
Proof.
  induction fuel as [ | fuel IH]; intros loop found; simpl;
    [ discriminate | ].
  destruct (masks_eqb (next loop) loop) eqn:Heq; [ | apply IH ].
  intros [= <-]. apply masks_eqb_eq, Heq.
Qed.

(* A mask found among those made for the handlers is the one made for
   some handler of that name. *)
Lemma lookup_handler_masks
    (make : continuation -> continuation_handler -> option (list bool))
    (k : continuation)
    (handlers : list (continuation * continuation_handler))
    (found : option (list bool)) :
  lookup k (map (fun '(k, h) => (k, make k h)) handlers) = Some found ->
  ∃ h : continuation_handler, found = make k h.
Proof.
  induction handlers as [ | [k' h] handlers IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb_spec k k') as [-> | _];
    [ intros [= <-]; exists h; reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* The pass relates a term to its result                                    *)
(* ======================================================================== *)

Fixpoint drop_dead_binds_statics (e : expr) :
  ∀ known : masks, binds_statics (drop_dead known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros known; cbn [drop_dead]; try reflexivity.
  - destruct (var_free_b x (drop_dead known body) || negb (named_may_go d));
      cbn [binds_statics]; apply drop_dead_binds_statics.
  - cbn [binds_statics]. rewrite drop_dead_binds_statics. reflexivity.
  - cbn [binds_statics handler_binds_statics].
    rewrite !drop_dead_binds_statics. reflexivity.
  - cbn [binds_statics]. rewrite drop_dead_binds_statics. f_equal.
    match goal with
    | |- context [settle ?fuel ?next ?first] =>
      destruct (settle fuel next first) as [loop | ]
    end;
      generalize (Let_cont_recursive handlers body); intros whole;
      match goal with
      | |- context [drop_dead_in_loop (?loop ++ known)] =>
        generalize (loop ++ known)
      end;
      intros inside;
      (induction handlers as [ | [k [params metadata handler]] handlers IH];
       [ reflexivity | ]);
      cbn [map existsb drop_dead_in_loop handler_binds_statics];
      rewrite drop_dead_binds_statics, IH; reflexivity.
Qed.

Fixpoint drop_dead_drops (e : expr) :
  ∀ known : masks, safe known e -> drops known e (drop_dead known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros known Hsafe; cbn [drop_dead].
  - destruct (var_free_b x (drop_dead known body) || negb (named_may_go d))
      eqn:Hkeep.
    + apply Drops_let, drop_dead_drops. exact Hsafe.
    + apply orb_false_iff in Hkeep as [Hfree Hgo].
      apply negb_false_iff in Hgo.
      apply Drops_let_unused; [ | apply drop_dead_drops; exact Hsafe | ].
      * destruct d as [s | [p | p | p | q | rp] args | | | | | ];
          try reflexivity.
        discriminate Hgo.
      * intros Hin. apply var_free_b_sound in Hin. congruence.
  - apply Drops_let_static, drop_dead_drops. exact Hsafe.
  - apply Drops_let_cont.
    + apply drop_dead_drops. intros k1 mask Hmask.
      specialize (Hsafe k1 mask Hmask). simpl in Hsafe.
      apply orb_false_iff in Hsafe as [Hhandler _]. exact Hhandler.
    + destruct (returned_to k body); [ exact I | apply used_params_fit ].
    + apply keep_kinds_kinds.
    + apply drop_dead_drops. intros k1 mask Hmask.
      destruct (Pos.eq_dec k1 k) as [-> | Hne].
      * rewrite mask_of_head in Hmask.
        destruct (returned_to k body); [ discriminate Hmask | reflexivity ].
      * rewrite mask_of_other in Hmask by exact Hne.
        specialize (Hsafe k1 mask Hmask). simpl in Hsafe.
        apply orb_false_iff in Hsafe as [_ Hbody]. exact Hbody.
  - set (whole := Let_cont_recursive handlers body) in *.
    (* Whatever masks one more round leaves as they are will do. *)
    assert (Hgeneral :
      ∀ (kept : continuation -> bool) (loop : masks),
        map (fun '(k, h) => (k, live_params (loop ++ known) (kept k) h))
          handlers
        = loop ->
        (∀ k : continuation, kept k = false -> returned_to k whole = false) ->
        drops known whole
          (Let_cont_recursive
             (map
                (fun '(k, h) =>
                   (k, drop_dead_in_loop (loop ++ known) (kept k) h))
                handlers)
             (drop_dead (loop ++ known) body))).
    { intros kept loop Hloop Hkept.
      assert (Hsafe' :
        ∀ (k : continuation) (mask : list bool),
          mask_of (loop ++ known) k = Some mask ->
          existsb (fun '(_, h) => returned_to_in_handler k h) handlers = false
          ∧ returned_to k body = false).
      { intros k mask. rewrite mask_of_app.
        destruct (lookup k loop) as [found | ] eqn:Hlookup.
        - intros ->. rewrite <- Hloop in Hlookup.
          destruct
            (lookup_handler_masks
               (fun k h => live_params (loop ++ known) (kept k) h) k handlers
               _ Hlookup)
            as [h Hfound].
          assert (Hk : kept k = false).
          { destruct (kept k); [ | reflexivity ].
            destruct h; discriminate Hfound. }
          specialize (Hkept k Hk). simpl in Hkept.
          apply orb_false_iff in Hkept. exact Hkept.
        - intros Hmask. specialize (Hsafe k mask Hmask). simpl in Hsafe.
          apply orb_false_iff in Hsafe. exact Hsafe. }
      apply (Drops_let_cont_recursive known loop).
      - clear Hsafe Hkept. revert Hsafe'.
        remember (loop ++ known) as inside eqn:Hinside. clear Hinside.
        rewrite <- Hloop. clear Hloop loop. clear whole.
        intros Hsafe'.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          [ apply Handlers_drop_nil | ].
        cbn [map live_params drop_dead_in_loop].
        assert (Hhandler : drops inside handler (drop_dead inside handler)).
        { apply drop_dead_drops. intros k1 mask Hmask.
          destruct (Hsafe' k1 mask Hmask) as [Hhandlers _].
          simpl in Hhandlers.
          apply orb_false_iff in Hhandlers as [Hhandler _]. exact Hhandler. }
        apply (Handlers_drop_cons inside k
          (if kept k then None
           else
             used_params (hm_is_exn_handler metadata) params
               (drop_dead inside handler))
          params metadata
          (keep_kinds
             (if kept k then None
              else
                used_params (hm_is_exn_handler metadata) params
                  (drop_dead inside handler))
             metadata)
          handler (drop_dead inside handler)).
        + exact Hhandler.
        + destruct (kept k); [ exact I | apply used_params_fit ].
        + apply keep_kinds_kinds.
        + apply IH. intros k1 mask Hmask.
          destruct (Hsafe' k1 mask Hmask) as [Hhandlers Hbody].
          simpl in Hhandlers.
          apply orb_false_iff in Hhandlers as [_ Hrest].
          exact (conj Hrest Hbody).
      - clear. unfold handlers_bind_statics. generalize (loop ++ known).
        intros inside.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          [ reflexivity | ].
        cbn [map existsb drop_dead_in_loop handler_binds_statics].
        rewrite drop_dead_binds_statics, IH. reflexivity.
      - apply drop_dead_drops. intros k mask Hmask.
        exact (proj2 (Hsafe' k mask Hmask)). }
    match goal with
    | |- context [settle ?fuel ?next ?first] =>
      destruct (settle fuel next first) as [loop | ] eqn:Hsettled
    end.
    + apply settle_fixed in Hsettled.
      apply (Hgeneral (fun k => returned_to k whole) loop Hsettled).
      intros k Hk. exact Hk.
    + apply (Hgeneral (fun _ => true)
        (map (fun '(k, _) => (k, None)) handlers)).
      * clear. generalize (map (fun '(k, _) => (k, None)) handlers ++ known).
        intros inside.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          [ reflexivity | ].
        cbn [map live_params]. rewrite IH. reflexivity.
      * intros k Hk. discriminate Hk.
  - apply Drops_apply.
    + destruct (mask_of known (ap_continuation ap)) as [mask | ] eqn:Hmask;
        [ | reflexivity ].
      specialize (Hsafe _ mask Hmask). simpl in Hsafe.
      rewrite Pos.eqb_refl in Hsafe. discriminate Hsafe.
    + destruct (mask_of known (ap_exn_continuation ap)) as [mask | ]
        eqn:Hmask; [ | reflexivity ].
      specialize (Hsafe _ mask Hmask). simpl in Hsafe.
      rewrite Pos.eqb_refl, orb_true_r in Hsafe. discriminate Hsafe.
  - apply Drops_apply_cont.
  - apply Drops_switch.
  - apply Drops_invalid.
Qed.

Theorem drop_dead_params_refines : drop_dead_params_refines_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (rounds_refines state_drops drops_round).
  apply (State_drops []);
    [ apply drop_dead_drops; intros k mask Hmask; discriminate Hmask
    | reflexivity | intros k; apply Same_drops ].
Qed.
